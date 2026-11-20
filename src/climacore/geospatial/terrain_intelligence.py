"""Terrain and Geospatial Exposure Intelligence Module for Clima-Core.
Provides terrain analytics and critical civil infrastructure vulnerability screening:

1. Digital Elevation Model (DEM) Topographic Processing:
   - 2D surface slope gradient (radians, degrees, percentage).
   - Topographic Wetness Index (TWI = ln(a / tan(beta))) (Beven & Kirkby, 1979).
   - Planform and profile curvature (concavity / pooling zones).
2. HydroRIVERS Drainage Network Topology:
   - Strahler stream order classification and upstream catchment area integration.
   - Distance-to-river reach buffer hazard calculation.
3. Civil Lifeline Critical Infrastructure Vulnerability:
   - Criticality-weighted asset exposure scoring (hospitals, schools, bridges, transit).
   - Composite Terrain & Infrastructure Vulnerability Index (TIVI).

References:
  - Beven, K. J., & Kirkby, M. J. (1979). A physically based, variable contributing area model
    of basin hydrology. Hydrological Sciences Bulletin, 24(1), 43-69.
  - Lehner, B., & Grill, G. (2013). Global river hydrography and network routing database.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


def compute_slope_and_aspect(
    elevation_grid: np.ndarray,
    cell_size_m: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Calculates local slope gradient, slope percentage, and aspect from a 2D elevation grid.
    
    Uses central differences for interior grid cells and one-sided differences at boundaries.
    
    Parameters:
      elevation_grid: 2D numpy array of elevations in meters (z).
      cell_size_m: Horizontal grid resolution in meters (dx = dy).
      
    Returns:
      Dictionary containing:
        - 'slope_radians': Local slope angle in radians.
        - 'slope_degrees': Local slope angle in degrees [0, 90].
        - 'slope_percent': Slope gradient percentage (100 * tan(beta)).
        - 'aspect_degrees': Compass azimuth of steepest descent [0, 360).
    """
    z = np.asarray(elevation_grid, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Elevation grid must be 2D, got shape {z.shape}.")
    if z.shape[0] < 2 or z.shape[1] < 2:
        raise ValueError("Elevation grid must have at least 2x2 dimensions.")

    dx = float(cell_size_m)
    dy = float(cell_size_m)

    # Compute numerical gradients dz/dy (rows) and dz/dx (cols)
    dz_dy, dz_dx = np.gradient(z, dy, dx)

    # Slope gradient = sqrt((dz/dx)^2 + (dz/dy)^2)
    gradient_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_radians = np.arctan(gradient_magnitude)
    slope_degrees = np.degrees(slope_radians)
    slope_percent = gradient_magnitude * 100.0

    # Aspect calculation: angle of steepest descent
    # Convention: 0 = North, 90 = East, 180 = South, 270 = West
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0.0, aspect_deg + 360.0, aspect_deg)
    # Flat areas (gradient == 0) have aspect = -1
    aspect_deg = np.where(gradient_magnitude < 1e-6, -1.0, aspect_deg)

    return {
        "slope_radians": slope_radians,
        "slope_degrees": slope_degrees,
        "slope_percent": slope_percent,
        "aspect_degrees": aspect_deg,
    }


def compute_topographic_wetness_index(
    slope_radians: np.ndarray,
    specific_contributing_area_m: Union[float, np.ndarray] = 100.0,
    min_slope_rad: float = 0.005,
) -> np.ndarray:
    """Computes Beven & Kirkby (1979) Topographic Wetness Index (TWI).
    
    Formulation:
      TWI = ln( a / tan( max(beta, beta_min) ) )
      
    Where:
      - a: Specific upslope contributing area per unit contour width (meters).
      - beta: Local ground slope angle (radians).
      - beta_min: Lower bound threshold preventing division by zero on flat terrain.
      
    Returns:
      Numpy array of TWI values (higher values indicate concave runoff accumulation zones).
    """
    beta = np.asarray(slope_radians, dtype=np.float64)
    beta_clamped = np.maximum(beta, float(min_slope_rad))
    tan_beta = np.tan(beta_clamped)

    a = np.asarray(specific_contributing_area_m, dtype=np.float64)
    if np.any(a <= 0.0):
        a = np.maximum(a, 1.0)

    twi = np.log(a / tan_beta)
    return twi


class HydroRiverNetwork:
    """HydroSHEDS HydroRIVERS Reach Network Analyzer."""

    def __init__(self, reach_records: Optional[List[Dict[str, Any]]] = None):
        """Initializes with a list of reach attribute dictionaries."""
        self.reaches = reach_records or []

    def filter_by_strahler(self, min_order: int = 1, max_order: int = 10) -> List[Dict[str, Any]]:
        """Filters reaches within a Strahler stream order bracket."""
        return [
            r for r in self.reaches
            if min_order <= int(r.get("strahler_order", r.get("river_strahler_order", 1))) <= max_order
        ]

    def compute_corridor_hazard_score(self, distance_to_reach_m: float, strahler_order: int) -> float:
        """Computes riverine proximity hazard score (0 to 100) based on reach order and distance.
        
        Large rivers (Strahler 6-8) have wider floodplains (up to 2000m influence zone);
        tributaries (Strahler 1-2) have narrower flash zones (< 200m).
        """
        dist = max(0.0, float(distance_to_reach_m))
        order = max(1, min(8, int(strahler_order)))

        # Effective floodplain corridor width based on Strahler order
        # Order 1: 150m, Order 4: 600m, Order 8: 2500m
        influence_width = 150.0 * (1.5 ** (order - 1))

        if dist >= influence_width:
            return 0.0

        proximity_ratio = 1.0 - (dist / influence_width)
        order_weight = 0.5 + 0.5 * (order / 8.0)
        score = proximity_ratio * order_weight * 100.0
        return float(np.clip(score, 0.0, 100.0))


class AssetExposureAnalyzer:
    """Critical Civil Infrastructure and Population Exposure Evaluator."""

    DEFAULT_CRITICALITY_WEIGHTS: Dict[str, float] = {
        "hospitals": 5.0,        # Acute emergency lifeline, trauma care
        "schools": 3.0,          # Vulnerable pediatric population & evacuation centers
        "bridges": 4.5,          # Critical transport bottlenecks and hydraulic chokepoints
        "transit_stations": 4.0, # Evacuation mobility corridors (metro/rail)
        "drainage_canals": 2.5,  # Stormwater conveyance channels
    }

    def __init__(self, criticality_weights: Optional[Dict[str, float]] = None):
        self.weights = criticality_weights or self.DEFAULT_CRITICALITY_WEIGHTS

    def compute_critical_asset_score(self, asset_counts: Dict[str, int]) -> float:
        """Computes weighted critical lifeline asset density score."""
        score = 0.0
        for asset_type, count in asset_counts.items():
            w = self.weights.get(asset_type, 1.0)
            score += float(count) * w
        return score

    def compute_terrain_infrastructure_vulnerability_index(
        self,
        mean_twi: float,
        mean_slope_deg: float,
        asset_counts: Dict[str, int],
        population_density_sqkm: float,
        distance_to_primary_river_m: float = 1000.0,
        primary_river_strahler: int = 4,
    ) -> Dict[str, Any]:
        """Calculates composite Terrain and Infrastructure Vulnerability Index (TIVI) (0 to 100).
        
        Components:
          1. Topographic Wetness Index score (high TWI -> waterlogging accumulation)
          2. Slope low-gradient vulnerability (flat urban terrain < 1 deg impedes drainage)
          3. River corridor proximity hazard
          4. Critical civil lifelines exposure
          5. Population density exposure
        """
        # 1. TWI score (typically 4 to 16, scaled to 0-100)
        twi_val = max(4.0, min(18.0, float(mean_twi)))
        twi_score = ((twi_val - 4.0) / 14.0) * 100.0

        # 2. Low-slope penalty (flatter terrain -> higher pluvial stagnation)
        slope_val = max(0.1, min(15.0, float(mean_slope_deg)))
        slope_stagnation_score = max(0.0, 100.0 - (slope_val / 5.0) * 100.0)

        # 3. River corridor hazard
        net = HydroRiverNetwork()
        river_hazard_score = net.compute_corridor_hazard_score(
            distance_to_primary_river_m, primary_river_strahler
        )

        # 4. Critical asset density score (log-scaled)
        raw_asset_score = self.compute_critical_asset_score(asset_counts)
        asset_exposure_score = min(100.0, (np.log1p(raw_asset_score) / np.log1p(1000.0)) * 100.0)

        # 5. Population exposure (log-scaled, 500 to 50,000 /km2)
        pop_dens = max(100.0, float(population_density_sqkm))
        pop_score = min(100.0, (np.log10(pop_dens) - 2.0) / 2.7 * 100.0)

        # Weighted composite score
        # Weights: TWI (25%), Slope Stagnation (20%), River Proximity (20%), Assets (20%), Population (15%)
        composite_tivi = (
            0.25 * twi_score +
            0.20 * slope_stagnation_score +
            0.20 * river_hazard_score +
            0.20 * asset_exposure_score +
            0.15 * pop_score
        )
        composite_tivi = float(np.clip(composite_tivi, 0.0, 100.0))

        if composite_tivi >= 75.0:
            tier = "VERY_HIGH"
        elif composite_tivi >= 55.0:
            tier = "HIGH"
        elif composite_tivi >= 35.0:
            tier = "MODERATE"
        else:
            tier = "LOW"

        return {
            "terrain_infrastructure_vulnerability_index": round(composite_tivi, 2),
            "vulnerability_tier": tier,
            "sub_indices": {
                "topographic_wetness_score": round(twi_score, 2),
                "slope_stagnation_score": round(slope_stagnation_score, 2),
                "river_corridor_hazard_score": round(river_hazard_score, 2),
                "critical_asset_exposure_score": round(asset_exposure_score, 2),
                "population_exposure_score": round(pop_score, 2),
            },
            "parameters": {
                "input_twi": round(mean_twi, 2),
                "input_slope_deg": round(mean_slope_deg, 2),
                "distance_to_river_m": round(distance_to_primary_river_m, 1),
                "river_strahler": primary_river_strahler,
            },
        }
