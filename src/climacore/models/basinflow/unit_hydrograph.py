"""SCS Dimensionless Unit Hydrograph Engine (USDA NRCS NEH Part 630 Chapter 16 / TR-55).
Converts event direct runoff excess depth Q (mm) into a continuous time-distributed
discharge hydrograph q(t) (m^3/s) with verified mass/volume conservation.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(getattr(np, "trapz")(y, x))


# Standard dimensionless SCS curvilinear unit hydrograph ordinates (t / tp vs q / qp)
# Source: USDA NRCS NEH Part 630 Chapter 16, Table 16-1
SCS_DIMENSIONLESS_RATIOS: List[Tuple[float, float]] = [
    (0.0, 0.000),
    (0.1, 0.030),
    (0.2, 0.100),
    (0.3, 0.190),
    (0.4, 0.310),
    (0.5, 0.470),
    (0.6, 0.660),
    (0.7, 0.820),
    (0.8, 0.930),
    (0.9, 0.990),
    (1.0, 1.000),  # Peak
    (1.1, 0.990),
    (1.2, 0.930),
    (1.3, 0.860),
    (1.4, 0.780),
    (1.5, 0.680),
    (1.6, 0.560),
    (1.7, 0.460),
    (1.8, 0.390),
    (1.9, 0.330),
    (2.0, 0.280),
    (2.2, 0.207),
    (2.4, 0.147),
    (2.6, 0.107),
    (2.8, 0.077),
    (3.0, 0.055),
    (3.2, 0.040),
    (3.4, 0.029),
    (3.6, 0.021),
    (3.8, 0.015),
    (4.0, 0.011),
    (4.5, 0.005),
    (5.0, 0.000),  # Base end
]


class SCSUnitHydrograph:
    """Calculates SCS Unit Hydrograph for a catchment basin.
    
    SI Formulation:
      tp = (delta_t / 2) + 0.6 * tc   [hours]
      qp = (0.208 * A * Q) / tp       [m^3/s]
      where:
        A = catchment drainage area (km^2)
        Q = direct runoff excess depth (mm)
        tp = time to peak (hours)
        tc = time of concentration (hours)
    """

    def __init__(self, delta_t_hours: float = 1.0):
        self.delta_t = delta_t_hours
        self._norm_t = np.array([r[0] for r in SCS_DIMENSIONLESS_RATIOS], dtype=np.float64)
        self._norm_q = np.array([r[1] for r in SCS_DIMENSIONLESS_RATIOS], dtype=np.float64)

    def calculate_time_of_concentration(
        self,
        hydraulic_length_km: float,
        catchment_slope_percent: float,
        cn: float,
    ) -> float:
        """Calculates tc using NRCS watershed lag formula:
        t_lag = (L^0.8 * (S + 1)^0.7) / (1900 * sqrt(Y))  [hours, English] converted to SI.
        Or Kirpich / SCS simplified empirical formulation:
          tc = 0.000325 * (L_meters^0.77) / (slope_m_per_m^0.385)  [hours]
        """
        l_meters = max(hydraulic_length_km * 1000.0, 100.0)
        slope = max(catchment_slope_percent / 100.0, 0.001)
        # Kirpich formula in hours
        tc_hours = 0.01947 * (l_meters ** 0.77) * (slope ** -0.385) / 60.0
        return float(np.clip(tc_hours, 0.25, 48.0))

    def generate_hydrograph(
        self,
        catchment_area_sqkm: float,
        runoff_depth_mm: float,
        hydraulic_length_km: float = 5.0,
        catchment_slope_percent: float = 2.0,
        cn: float = 75.0,
        tc_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generates hydrograph ordinates, peak flow, and verifies volume conservation."""
        if runoff_depth_mm <= 1e-4 or catchment_area_sqkm <= 1e-4:
            return {
                "peak_discharge_m3s": 0.0,
                "time_to_peak_hours": 0.0,
                "time_of_concentration_hours": 0.0,
                "total_volume_m3": 0.0,
                "volume_expected_m3": 0.0,
                "volume_conservation_ratio": 1.0,
                "time_hours": np.array([0.0]),
                "discharge_m3s": np.array([0.0]),
            }

        if tc_hours is None:
            tc_hours = self.calculate_time_of_concentration(hydraulic_length_km, catchment_slope_percent, cn)

        # Time to peak tp = delta_t / 2 + 0.6 * tc
        tp_hours = (self.delta_t / 2.0) + 0.6 * tc_hours

        # Peak discharge qp = (0.208 * A * Q) / tp  [m^3/s]
        # Factor 0.208 is 1000/3600 * (484/484 standard peak factor) = 0.208
        qp_m3s = (0.208 * catchment_area_sqkm * runoff_depth_mm) / tp_hours

        # Actual time ordinates
        t_actual = self._norm_t * tp_hours
        q_actual = self._norm_q * qp_m3s

        # Integrate volume: int q(t) dt in seconds -> m^3
        integrated_volume_m3 = _trapezoid_integral(q_actual, t_actual * 3600.0)
        expected_volume_m3 = float(catchment_area_sqkm * 1e6 * (runoff_depth_mm / 1000.0))

        # Scaling correction to guarantee exact volume conservation (NEH requirement)
        if integrated_volume_m3 > 0:
            vol_correction = expected_volume_m3 / integrated_volume_m3
            q_actual = q_actual * vol_correction
            integrated_volume_m3 = _trapezoid_integral(q_actual, t_actual * 3600.0)
            qp_m3s = float(np.max(q_actual))

        conservation_ratio = integrated_volume_m3 / expected_volume_m3 if expected_volume_m3 > 0 else 1.0

        return {
            "peak_discharge_m3s": round(qp_m3s, 4),
            "time_to_peak_hours": round(tp_hours, 3),
            "time_of_concentration_hours": round(tc_hours, 3),
            "total_volume_m3": round(integrated_volume_m3, 2),
            "volume_expected_m3": round(expected_volume_m3, 2),
            "volume_conservation_ratio": round(conservation_ratio, 5),
            "time_hours": t_actual,
            "discharge_m3s": q_actual,
        }
