"""
WHO Child Growth Standards — LMS z-score computation engine.
Source: WHO Multicentre Growth Reference Study (MGRS, 2006)
"""
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from typing import Tuple

# =============================================================================
# WHO LMS Reference Tables
# =============================================================================

WHO_LMS_TABLES = {
    'weight_for_age': {
        'male': {
            0: (0.3487, 3.3464, 0.14602), 1: (0.2297, 4.4709, 0.13395),
            2: (0.1970, 5.5675, 0.12385), 3: (0.1738, 6.3762, 0.11727),
            4: (0.1553, 7.0023, 0.11316), 5: (0.1395, 7.5105, 0.11080),
            6: (0.1257, 7.9340, 0.10958), 7: (0.1134, 8.2970, 0.10902),
            8: (0.1021, 8.6151, 0.10882), 9: (0.0917, 8.9014, 0.10881),
            10: (0.0820, 9.1649, 0.10891), 11: (0.0730, 9.4122, 0.10906),
            12: (0.0644, 9.6479, 0.10925), 15: (0.0407, 10.3002, 0.10986),
            18: (0.0198, 10.8985, 0.11080), 21: (0.0012, 11.4753, 0.11211),
            24: (-0.0155, 12.0515, 0.11374), 27: (-0.0305, 12.6371, 0.11559),
            30: (-0.0440, 13.2372, 0.11751), 33: (-0.0562, 13.8487, 0.11940),
            36: (-0.0672, 14.3340, 0.12110)
        },
        'female': {
            0: (0.3809, 3.2322, 0.14171), 1: (0.1714, 4.1873, 0.13724),
            2: (0.0962, 5.1282, 0.12950), 3: (0.0402, 5.8458, 0.12414),
            4: (-0.0050, 6.4237, 0.12052), 5: (-0.0430, 6.8985, 0.11802),
            6: (-0.0756, 7.2970, 0.11628), 7: (-0.1039, 7.6422, 0.11504),
            8: (-0.1288, 7.9487, 0.11410), 9: (-0.1507, 8.2254, 0.11334),
            10: (-0.1700, 8.4800, 0.11267), 11: (-0.1872, 8.7192, 0.11206),
            12: (-0.2024, 8.9481, 0.11150), 15: (-0.2388, 9.5826, 0.11036),
            18: (-0.2659, 10.1864, 0.10971), 21: (-0.2867, 10.7851, 0.10961),
            24: (-0.3030, 11.3962, 0.11003), 27: (-0.3157, 12.0262, 0.11091),
            30: (-0.3256, 12.6746, 0.11214), 33: (-0.3332, 13.3363, 0.11358),
            36: (-0.3390, 13.9244, 0.11500)
        }
    },
    'length_for_age': {
        'male': {
            0: (1.0, 49.8842, 0.03795), 1: (1.0, 54.7244, 0.03557),
            2: (1.0, 58.4249, 0.03424), 3: (1.0, 61.4292, 0.03328),
            4: (1.0, 63.8860, 0.03257), 5: (1.0, 65.9026, 0.03204),
            6: (1.0, 67.6236, 0.03165), 7: (1.0, 69.1645, 0.03139),
            8: (1.0, 70.5994, 0.03124), 9: (1.0, 71.9687, 0.03117),
            10: (1.0, 73.2812, 0.03118), 11: (1.0, 74.5388, 0.03126),
            12: (1.0, 75.7488, 0.03141), 15: (1.0, 79.2550, 0.03209),
            18: (1.0, 82.4486, 0.03314), 21: (1.0, 85.3670, 0.03441),
            24: (1.0, 87.8161, 0.03580), 27: (1.0, 90.3521, 0.03706),
            30: (1.0, 92.6900, 0.03840), 33: (1.0, 94.8476, 0.03978),
            36: (1.0, 96.0700, 0.04100)
        },
        'female': {
            0: (1.0, 49.1477, 0.03790), 1: (1.0, 53.6872, 0.03598),
            2: (1.0, 57.0673, 0.03468), 3: (1.0, 59.8029, 0.03374),
            4: (1.0, 62.0899, 0.03307), 5: (1.0, 64.0301, 0.03261),
            6: (1.0, 65.7311, 0.03228), 7: (1.0, 67.2873, 0.03204),
            8: (1.0, 68.7498, 0.03189), 9: (1.0, 70.1435, 0.03183),
            10: (1.0, 71.4818, 0.03183), 11: (1.0, 72.7710, 0.03190),
            12: (1.0, 74.0150, 0.03204), 15: (1.0, 77.5049, 0.03270),
            18: (1.0, 80.7128, 0.03378), 21: (1.0, 83.6593, 0.03510),
            24: (1.0, 86.4160, 0.03650), 27: (1.0, 88.6254, 0.03790),
            30: (1.0, 91.2920, 0.03930), 33: (1.0, 93.4137, 0.04070),
            36: (1.0, 95.4036, 0.04210)
        }
    },
    'head_circumference_for_age': {
        'male': {
            0: (1.0, 34.4618, 0.03686), 1: (1.0, 37.2759, 0.03133),
            2: (1.0, 39.1285, 0.02997), 3: (1.0, 40.5135, 0.02918),
            4: (1.0, 41.6317, 0.02868), 5: (1.0, 42.5576, 0.02837),
            6: (1.0, 43.3306, 0.02817), 7: (1.0, 43.9803, 0.02804),
            8: (1.0, 44.5300, 0.02796), 9: (1.0, 44.9998, 0.02792),
            10: (1.0, 45.4051, 0.02790), 11: (1.0, 45.7573, 0.02790),
            12: (1.0, 46.0661, 0.02791), 15: (1.0, 46.8032, 0.02798),
            18: (1.0, 47.3677, 0.02809), 21: (1.0, 47.8109, 0.02824),
            24: (1.0, 48.1656, 0.02841), 36: (1.0, 49.1318, 0.02902)
        },
        'female': {
            0: (1.0, 33.8787, 0.03496), 1: (1.0, 36.5463, 0.03187),
            2: (1.0, 38.2521, 0.03069), 3: (1.0, 39.5328, 0.02997),
            4: (1.0, 40.5817, 0.02950), 5: (1.0, 41.4590, 0.02917),
            6: (1.0, 42.1995, 0.02894), 7: (1.0, 42.8290, 0.02878),
            8: (1.0, 43.3670, 0.02866), 9: (1.0, 43.8300, 0.02858),
            10: (1.0, 44.2319, 0.02854), 11: (1.0, 44.5844, 0.02851),
            12: (1.0, 44.8965, 0.02851), 15: (1.0, 45.6029, 0.02856),
            18: (1.0, 46.1500, 0.02868), 21: (1.0, 46.5791, 0.02884),
            24: (1.0, 46.9232, 0.02904), 36: (1.0, 47.8600, 0.02960)
        }
    }
}


class WHOZScoreEngine:
    """WHO Child Growth Standards z-score computation engine using LMS method."""

    def __init__(self, lms_tables: dict = None):
        self.lms_tables = lms_tables or WHO_LMS_TABLES
        self._interpolators = {}

    def _get_interpolated_lms(self, metric: str, sex: str,
                               age_months: float) -> Tuple[float, float, float]:
        cache_key = (metric, sex)
        if cache_key not in self._interpolators:
            table = self.lms_tables[metric][sex]
            ages = sorted(table.keys())
            L_vals = [table[a][0] for a in ages]
            M_vals = [table[a][1] for a in ages]
            S_vals = [table[a][2] for a in ages]
            self._interpolators[cache_key] = {
                'L': interp1d(ages, L_vals, kind='linear', fill_value='extrapolate'),
                'M': interp1d(ages, M_vals, kind='linear', fill_value='extrapolate'),
                'S': interp1d(ages, S_vals, kind='linear', fill_value='extrapolate'),
            }
        interps = self._interpolators[cache_key]
        return (float(interps['L'](age_months)),
                float(interps['M'](age_months)),
                float(interps['S'](age_months)))

    def compute_zscore(self, metric: str, sex: str,
                       age_months: float, value: float) -> float:
        L, M, S = self._get_interpolated_lms(metric, sex, age_months)
        if value <= 0 or M <= 0:
            return 0.0
        if abs(L) > 0.001:
            z = ((value / M) ** L - 1) / (L * S)
        else:
            z = np.log(value / M) / S
        return float(np.clip(z, -10, 10))

    def zscore_to_value(self, metric: str, sex: str,
                        age_months: float, z: float) -> float:
        L, M, S = self._get_interpolated_lms(metric, sex, age_months)
        if abs(L) > 0.001:
            inner = 1 + L * S * z
            if inner <= 0:
                inner = 0.001
            value = M * (inner ** (1.0 / L))
        else:
            value = M * np.exp(S * z)
        return max(float(value), 0.0)

    def zscore_to_percentile(self, z: float) -> float:
        return float(stats.norm.cdf(z) * 100)

    def get_percentile_value(self, metric: str, sex: str,
                             age_months: float, percentile: float) -> float:
        z = stats.norm.ppf(percentile / 100.0)
        return self.zscore_to_value(metric, sex, age_months, z)

    def get_median(self, metric: str, sex: str, age_months: float) -> float:
        _, M, _ = self._get_interpolated_lms(metric, sex, age_months)
        return float(M)

    @property
    def available_metrics(self) -> list:
        return list(self.lms_tables.keys())
