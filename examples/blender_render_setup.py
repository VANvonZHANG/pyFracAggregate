"""
Example: Blender Rendering Setup
--------------------------------
This is an example script intended to be run INSIDE Blender.
It assumes you have already loaded the aggregate (e.g. by running the generated load_aggregate.py script).
This script will automatically set up the camera, lighting, and render an image.
"""
import bpy
import numpy as np
import os

def setup_lighting_and_camera(collection_name="Soot_Cluster_Final"):
    # Find the collection
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        print(f"Collection {collection_name} not found!")
        return

    # Calculate bounding box
    positions = []
    for obj in collection.objects:
        if obj.type == 'MESH':
            positions.append(list(obj.location))
            
    if not positions:
        print("No mesh objects found in collection.")
        return
        
    p_arr = np.array(positions)
    center = p_arr.mean(axis=0)
    max_dim = np.max(np.linalg.norm(p_arr - center, axis=1)) + 2.0
    
    # Add Camera
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Position camera
    cam_dist = max_dim * 3.5
    cam_obj.location = (center[0] + cam_dist, center[1] + cam_dist, center[2] + cam_dist * 0.5)
    
    import mathutils
    # Point camera towards center
    direction = mathutils.Vector(center - cam_obj.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_mode = 'QUATERNION'
    cam_obj.rotation_quaternion = rot_quat

    # Add Sun Light
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (center[0] + 10, center[1] + 10, center[2] + 20)
    light_data.energy = 5.0
    
    # Add Area Light for softer shadows
    area_data = bpy.data.lights.new(name="Area", type='AREA')
    area_obj = bpy.data.objects.new(name="Area", object_data=area_data)
    bpy.context.scene.collection.objects.link(area_obj)
    area_obj.location = (center[0] - 10, center[1] - 10, center[2] + 15)
    area_data.energy = 200.0
    area_data.size = 5.0
    
    print("Camera and lights setup successfully.")

def render_scene(output_path="render.png"):
    bpy.context.scene.render.filepath = os.path.abspath(output_path)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
    print(f"Render saved to: {bpy.context.scene.render.filepath}")

if __name__ == "__main__":
    setup_lighting_and_camera(collection_name="Soot_Cluster_Final")
    
    # Uncomment the next line to actually trigger a render in Blender
    # render_scene("my_render.png") 
