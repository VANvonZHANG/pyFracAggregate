import numpy as np
import yaml
from pyFracAggregate.core.aggregate import Aggregate


def _to_native(obj):
    """Recursively convert NumPy types to native Python types for YAML serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def export_yaml(
    aggregate: Aggregate,
    output_path: str,
    *,
    generation_params: dict | None = None,
    analysis_results: dict | None = None,
) -> None:
    """Export aggregate to YAML with optional generation params and analysis results.

    Args:
        aggregate: The fractal aggregate object to export.
        output_path: Path to save the YAML file.
        generation_params: Optional dict of generation parameters (method, df, kf, etc.).
        analysis_results: Optional dict of analysis results (Rg, center_of_mass, etc.).
    """
    data = {}

    if generation_params is not None:
        data["generation"] = _to_native(generation_params)

    data["aggregate"] = _to_native({
        "n_particles": aggregate.current_size,
        "length_unit": aggregate.length_unit,
        "mass_unit": aggregate.mass_unit,
        "density": aggregate.density,
        "positions": aggregate.positions,
        "radii": aggregate.radii,
        "masses": aggregate.masses,
    })

    if analysis_results is not None:
        data["analysis"] = _to_native(analysis_results)

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
