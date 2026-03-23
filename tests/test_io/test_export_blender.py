import pytest
import os
import json
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.io.export_blender import export_to_json, generate_blender_script

def test_export_to_json_and_script(tmp_path):
    # Use pytest's tmp_path to provide a temporary directory
    agg = Aggregate(3)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    agg.add_particle(2.0, 0.0, 0.0, 1.0, 1.0)
    agg.add_particle(1.0, 1.732, 0.0, 1.0, 1.0)
    
    json_path = tmp_path / "test_aggregate.json"
    script_path = tmp_path / "load_in_blender.py"
    
    # 1. Test exporting to JSON
    export_to_json(agg, str(json_path))
    assert os.path.exists(json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    assert data["num_particles"] == 3
    assert len(data["particles"]) == 3
    assert data["particles"][1]["x"] == 2.0
    
    # 2. Test generating Blender script
    generate_blender_script(str(json_path), str(script_path))
    assert os.path.exists(script_path)
    
    with open(script_path, 'r') as f:
        script_content = f.read()
        
    # Verify that the path in the generated script is correctly embedded and escaped
    abs_json_path = os.path.abspath(str(json_path)).replace('\\', '/')
    assert f'DATA_PATH = "{abs_json_path}"' in script_content
    assert "GeometryNodeInstanceOnPoints" in script_content
