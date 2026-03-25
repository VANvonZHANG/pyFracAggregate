import json
from pyFracAggregate.core.aggregate import Aggregate

def export_to_json(aggregate: Aggregate, output_path: str) -> None:
    """
    Exports the aggregate structure to a standard JSON file.
    
    Args:
        aggregate (Aggregate): The fractal aggregate object to export.
        output_path (str): The path to save the generated JSON file.
    """
    data = {
        "num_particles": aggregate.current_size,
        "length_unit": aggregate.length_unit,
        "mass_unit": aggregate.mass_unit,
        "density": aggregate.density,
        "positions": aggregate.positions.tolist(),
        "radii": aggregate.radii.tolist(),
        "masses": aggregate.masses.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Data successfully exported to JSON: {output_path}")
