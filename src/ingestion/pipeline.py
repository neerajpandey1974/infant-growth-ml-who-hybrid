"""
NHANES Data Ingestion Pipeline.
Downloads .XPT files from CDC, parses them, and transforms into training data.
"""
import os
import ssl
import time
import warnings
import urllib.request
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd

from config.settings import (
    NHANES_CYCLES, CYCLE_URL_MAP, DEMO_SUFFIX, BMX_SUFFIX,
    RAW_DIR, CACHE_DIR, MAX_AGE_MONTHS,
    DOWNLOAD_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
)

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _download_xpt(url: str, cache_path: Path) -> pd.DataFrame:
    """Download and read a .XPT file with retry logic and SSL fallback."""
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        return pd.read_sas(str(cache_path), format='xport', encoding='utf-8')

    for attempt in range(MAX_RETRIES):
        try:
            # Try requests first
            import requests
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, verify=False)
            if resp.status_code == 200 and len(resp.content) > 1000:
                if b'<html' not in resp.content[:500].lower():
                    cache_path.write_bytes(resp.content)
                    return pd.read_sas(BytesIO(resp.content), format='xport',
                                       encoding='utf-8')
        except Exception:
            pass

        try:
            # SSL fallback
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT,
                                        context=ctx) as resp:
                data = resp.read()
                if len(data) > 1000 and b'<html' not in data[:500].lower():
                    cache_path.write_bytes(data)
                    return pd.read_sas(BytesIO(data), format='xport',
                                       encoding='utf-8')
        except Exception:
            pass

        try:
            # pandas direct read fallback
            return pd.read_sas(url, format='xport', encoding='utf-8')
        except Exception:
            pass

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts")


def download_all_cycles(cycles: list = None) -> dict:
    """Download NHANES data for all specified cycles.

    Returns dict of {cycle: {'demo': DataFrame, 'bmx': DataFrame}}
    """
    cycles = cycles or NHANES_CYCLES
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for cycle in cycles:
        base_url = CYCLE_URL_MAP.get(cycle)
        if not base_url:
            continue

        demo_code = DEMO_SUFFIX.get(cycle)
        bmx_code = BMX_SUFFIX.get(cycle)

        if not demo_code or not bmx_code:
            continue

        demo_url = f"{base_url}/{demo_code}.XPT"
        bmx_url = f"{base_url}/{bmx_code}.XPT"

        try:
            demo_df = _download_xpt(
                demo_url, CACHE_DIR / f"{demo_code}.XPT"
            )
            bmx_df = _download_xpt(
                bmx_url, CACHE_DIR / f"{bmx_code}.XPT"
            )
            results[cycle] = {'demo': demo_df, 'bmx': bmx_df}
            print(f"  ✓ {cycle}: DEMO={len(demo_df)}, BMX={len(bmx_df)}")
        except Exception as e:
            print(f"  ✗ {cycle}: {e}")

    return results


def transform_cycle(demo_df: pd.DataFrame, bmx_df: pd.DataFrame,
                    cycle: str) -> pd.DataFrame:
    """Transform a single NHANES cycle into the training format."""

    # ── Age extraction ──
    age_col = None
    for col in ['RIDEXAGM', 'RIDAGEEX']:
        if col in demo_df.columns:
            age_col = col
            break

    if age_col:
        demo_df['age_months'] = pd.to_numeric(
            demo_df[age_col], errors='coerce'
        )
    elif 'RIDAGEMN' in demo_df.columns:
        demo_df['age_months'] = pd.to_numeric(
            demo_df['RIDAGEMN'], errors='coerce'
        )
    else:
        demo_df['age_months'] = pd.to_numeric(
            demo_df.get('RIDAGEYR', pd.Series(dtype=float)), errors='coerce'
        ) * 12

    # ── Filter infants 0–36 months ──
    demo_df = demo_df[
        demo_df['age_months'].notna() &
        (demo_df['age_months'] >= 0) &
        (demo_df['age_months'] <= MAX_AGE_MONTHS)
    ].copy()

    if len(demo_df) == 0:
        return pd.DataFrame()

    # ── Demographics ──
    demo_df['sex'] = demo_df['RIAGENDR'].map({1: 'male', 2: 'female'})
    demo_df['sex_female'] = (demo_df['sex'] == 'female').astype(float)

    if 'RIDRETH1' in demo_df.columns:
        demo_df['race_eth'] = demo_df['RIDRETH1']
    elif 'RIDRETH3' in demo_df.columns:
        demo_df['race_eth'] = demo_df['RIDRETH3']
    else:
        demo_df['race_eth'] = np.nan

    if 'INDFMPIR' in demo_df.columns:
        demo_df['income_poverty_ratio'] = pd.to_numeric(
            demo_df['INDFMPIR'], errors='coerce'
        )
    else:
        demo_df['income_poverty_ratio'] = np.nan

    if 'DMDFMSIZ' in demo_df.columns:
        demo_df['family_size'] = pd.to_numeric(
            demo_df['DMDFMSIZ'], errors='coerce'
        )
    else:
        demo_df['family_size'] = np.nan

    # ── Body measures ──
    bmx_rename = {
        'BMXWT': 'weight_kg',
        'BMXRECUM': 'length_cm',
        'BMXHEAD': 'head_circ_cm',
        'BMXARMC': 'arm_circ_cm',
        'BMXARML': 'arm_length_cm',
    }
    for old, new in bmx_rename.items():
        if old in bmx_df.columns:
            bmx_df[new] = pd.to_numeric(bmx_df[old], errors='coerce')

    # ── Merge ──
    merged = demo_df.merge(bmx_df, on='SEQN', how='inner')

    # ── Birth weight from ECQ if available ──
    for col in ['WHD010', 'WHD020']:
        if col in merged.columns:
            bw = pd.to_numeric(merged[col], errors='coerce')
            if col == 'WHD020':
                bw = bw * 0.453592  # lbs to kg
            merged['birth_weight_kg'] = bw
            break
    if 'birth_weight_kg' not in merged.columns:
        merged['birth_weight_kg'] = np.nan

    # ── SEQN deduplication ──
    merged = merged.drop_duplicates(subset='SEQN', keep='first')

    # ── Select columns ──
    keep_cols = [
        'SEQN', 'age_months', 'sex', 'sex_female', 'race_eth',
        'income_poverty_ratio', 'family_size', 'birth_weight_kg',
        'weight_kg', 'length_cm', 'head_circ_cm', 'arm_circ_cm', 'arm_length_cm'
    ]
    available = [c for c in keep_cols if c in merged.columns]
    result = merged[available].copy()
    result['cycle'] = cycle

    return result


def build_nhanes_dataset(cycles: list = None) -> pd.DataFrame:
    """Full pipeline: download, parse, transform, merge all NHANES cycles."""
    print("=" * 60)
    print("NHANES Data Ingestion Pipeline")
    print("=" * 60)

    print("\n1. Downloading NHANES cycles...")
    raw_data = download_all_cycles(cycles)

    print(f"\n2. Transforming {len(raw_data)} cycles...")
    frames = []
    for cycle, data in raw_data.items():
        df = transform_cycle(data['demo'], data['bmx'], cycle)
        if len(df) > 0:
            frames.append(df)
            print(f"  ✓ {cycle}: {len(df)} infant records")

    if not frames:
        raise RuntimeError("No data produced from any NHANES cycle")

    combined = pd.concat(frames, ignore_index=True)

    # ── Outlier validation ──
    bounds = {
        'weight_kg': (0.5, 30), 'length_cm': (30, 120),
        'head_circ_cm': (25, 60), 'arm_circ_cm': (5, 30),
    }
    for col, (lo, hi) in bounds.items():
        if col in combined.columns:
            mask = combined[col].between(lo, hi) | combined[col].isna()
            n_out = (~mask).sum()
            if n_out > 0:
                combined.loc[~mask, col] = np.nan
                print(f"  Outliers removed: {col}: {n_out}")

    print(f"\n3. Final dataset: {len(combined)} records")
    print(f"   Columns: {list(combined.columns)}")
    return combined
