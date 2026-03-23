import json
import os
import textwrap
from pyFracAggregate.core.aggregate import Aggregate

def export_to_json(aggregate: Aggregate, filepath: str) -> None:
    """
    将团簇数据导出为 JSON 文件，用于与外部渲染器（如 Blender）进行轻量级解耦通信。
    
    Args:
        aggregate (Aggregate): 要导出的团簇对象。
        filepath (str): JSON 文件的输出路径。
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
    """
    生成一个可直接在 Blender 中运行的 Python 脚本。
    该脚本会读取指定的 JSON 文件，使用顶点实例化（或 Geometry Nodes）高效生成海量粒子。
    
    Args:
        json_filepath (str): 前置步骤生成的 JSON 数据文件绝对路径。
        output_script_path (str): 生成的 Blender Python 挂载脚本路径。
    """
    # 转换为绝对路径，避免在 Blender 执行时由于工作目录问题找不到文件
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
        
        # 1. 创建基础网格对象存储所有顶点位置
        mesh = bpy.data.meshes.new(name="Fractal_Mesh")
        obj = bpy.data.objects.new("Fractal_Aggregate", mesh)
        bpy.context.collection.objects.link(obj)
        
        # 将点坐标写入 mesh
        verts = [(p['x'], p['y'], p['z']) for p in particles]
        mesh.from_pydata(verts, [], [])
        mesh.update()
        
        # 将半径写入顶点属性 (自定义属性)
        radii = [p['r'] for p in particles]
        if 'radius' not in mesh.attributes:
            radius_attr = mesh.attributes.new(name='radius', type='FLOAT', domain='POINT')
        else:
            radius_attr = mesh.attributes['radius']
            
        radius_attr.data.foreach_set('value', radii)
        
        # 2. 为对象添加 Geometry Nodes 修饰器
        modifier = obj.modifiers.new(name="GeometryNodes", type='NODES')
        
        # 创建新的 Node Tree
        node_tree = bpy.data.node_groups.new(name="Fractal_Instance_Tree", type='GeometryNodeTree')
        modifier.node_group = node_tree
        
        # 清空默认节点
        node_tree.nodes.clear()
        
        # 构建节点网络
        # ----------------------------------------------------
        # Group Input
        node_input = node_tree.nodes.new('NodeGroupInput')
        node_tree.interface.new_socket(name="Geometry", in_out='IN', socket_type='NodeSocketGeometry')
        node_input.location = (-400, 0)
        
        # Instance on Points
        node_instance = node_tree.nodes.new('GeometryNodeInstanceOnPoints')
        node_instance.location = (0, 0)
        
        # Ico Sphere (作为基础粒子)
        node_sphere = node_tree.nodes.new('GeometryNodeMeshIcoSphere')
        node_sphere.inputs['Subdivisions'].default_value = 3
        node_sphere.inputs['Radius'].default_value = 1.0  # 基础半径 1.0
        node_sphere.location = (-200, -200)
        
        # Named Attribute (读取我们在 mesh 中存入的 'radius' 属性)
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
        
        # 链接节点
        links = node_tree.links
        links.new(node_input.outputs[0], node_instance.inputs['Points'])
        links.new(node_sphere.outputs['Mesh'], node_instance.inputs['Instance'])
        links.new(node_attr.outputs['Attribute'], node_instance.inputs['Scale'])
        links.new(node_instance.outputs['Instances'], node_smooth.inputs['Geometry'])
        links.new(node_smooth.outputs['Geometry'], node_output.inputs[0])
        
        # 将视角对准生成的物体
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.view3d.view_selected()
        except RuntimeError:
            pass # 可能在后台无 UI 模式下运行
            
        print("Fractal Aggregate successfully loaded into Blender Geometry Nodes!")

    if __name__ == "__main__":
        load_fractal_aggregate()
    """
    
    script_content = textwrap.dedent(script_content)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_script_path)) or '.', exist_ok=True)
    with open(output_script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
