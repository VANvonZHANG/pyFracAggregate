import json
import os
import textwrap
from pyFracAggregate.core.aggregate import Aggregate

def export_to_json(aggregate: Aggregate, filepath: str) -> None:
    """Exports cluster data to a JSON file for lightweight decoupled communication.

    Used for external renderers like Blender.
    
    Args:
        aggregate (Aggregate): The cluster object to export.
        filepath (str): Output path for the JSON file.
    """
    positions = aggregate.positions.tolist()
    radii = aggregate.radii.tolist()
    
    data = {
        "num_particles": aggregate.current_size,
        "particles": [
            {
                "x": float(positions[i][0]),
                "y": float(positions[i][1]),
                "z": float(positions[i][2]),
                "r": float(radii[i])
            }
            for i in range(aggregate.current_size)
        ]
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or '.', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def generate_blender_script(json_filepath: str, output_script_path: str) -> None:
    """Generates a Python script that can be run directly within Blender.

    The script reads a JSON file and uses vertex instancing (or Geometry Nodes) 
    to efficiently generate massive numbers of particles.
    
    Args:
        json_filepath (str): Absolute path to the JSON data file generated previously.
        output_script_path (str): Path for the generated Blender Python script.
    """
    # Convert to absolute path to avoid file-not-found issues in Blender due to working directory
    abs_json_path = os.path.abspath(json_filepath).replace('\\', '/')
    
    script_content = f"""\
    import bpy
    import json
    import os

    # JSON File path
    DATA_PATH = "{abs_json_path}"

    def load_fractal_aggregate():
        if not os.path.exists(DATA_PATH):
            print(f"Error: Could not find data file at {{DATA_PATH}}")
            return
            
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
            
        particles = data['particles']
        num_particles = data['num_particles']
        
        print(f"Loading {{num_particles}} particles from JSON...")
        
        # 1. Create base mesh object to store all vertex positions
        mesh = bpy.data.meshes.new(name="Fractal_Mesh")
        obj = bpy.data.objects.new("Fractal_Aggregate", mesh)
        bpy.context.collection.objects.link(obj)
        
        # Write point coordinates to mesh
        verts = [(p['x'], p['y'], p['z']) for p in particles]
        mesh.from_pydata(verts, [], [])
        mesh.update()
        
        # Write radii to vertex attributes (custom attribute)
        radii = [p['r'] for p in particles]
        if 'radius' not in mesh.attributes:
            radius_attr = mesh.attributes.new(name='radius', type='FLOAT', domain='POINT')
        else:
            radius_attr = mesh.attributes['radius']
            
        radius_attr.data.foreach_set('value', radii)
        
        # 2. Add Geometry Nodes modifier to the object
        modifier = obj.modifiers.new(name="GeometryNodes", type='NODES')
        
        # Create new Node Tree
        node_tree = bpy.data.node_groups.new(name="Fractal_Instance_Tree", type='GeometryNodeTree')
        modifier.node_group = node_tree
        
        # Clear default nodes
        node_tree.nodes.clear()
        
        # Build node network
        # ----------------------------------------------------
        # Group Input
        node_input = node_tree.nodes.new('NodeGroupInput')
        node_tree.interface.new_socket(name="Geometry", in_out='IN', socket_type='NodeSocketGeometry')
        node_input.location = (-400, 0)
        
        # Instance on Points
        node_instance = node_tree.nodes.new('GeometryNodeInstanceOnPoints')
        node_instance.location = (0, 0)
        
        # Ico Sphere (as base particle)
        node_sphere = node_tree.nodes.new('GeometryNodeMeshIcoSphere')
        node_sphere.inputs['Subdivisions'].default_value = 3
        node_sphere.inputs['Radius'].default_value = 1.0  # Base radius 1.0
        node_sphere.location = (-200, -200)
        
        # Named Attribute (reading the 'radius' attribute we stored in the mesh)
        node_attr = node_tree.nodes.new('GeometryNodeInputNamedAttribute')
        node_attr.data_type = 'FLOAT'
        node_attr.inputs['Name'].default_value = 'radius'
        node_attr.location = (-200, -400)
        
        # Set Shade Smooth
        node_smooth = node_tree.nodes.new('GeometryNodeSetShadeSmooth')
        node_smooth.location = (200, 0)
        
        # Group Output
        node_output = node_tree.nodes.new('NodeGroupOutput')
        node_tree.interface.new_socket(name="Geometry", in_out='OUT', socket_type='NodeSocketGeometry')
        node_output.location = (400, 0)
        
        # Link nodes
        links = node_tree.links
        links.new(node_input.outputs[0], node_instance.inputs['Points'])
        links.new(node_sphere.outputs['Mesh'], node_instance.inputs['Instance'])
        links.new(node_attr.outputs['Attribute'], node_instance.inputs['Scale'])
        links.new(node_instance.outputs['Instances'], node_smooth.inputs['Geometry'])
        links.new(node_smooth.outputs['Geometry'], node_output.inputs[0])
        
        # Center view on generated object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.view3d.view_selected()
        except RuntimeError:
            pass # Might be running in background/headless mode
            
        print("Fractal Aggregate successfully loaded into Blender Geometry Nodes!")

    if __name__ == "__main__":
        load_fractal_aggregate()
    """
    
    script_content = textwrap.dedent(script_content)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_script_path)) or '.', exist_ok=True)
    with open(output_script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
