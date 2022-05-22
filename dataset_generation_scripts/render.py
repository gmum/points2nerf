# Modified: https://github.com/panmari/stanford-shapenet-renderer
# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Tested with Blender 2.9
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os, math, re
import bpy
from glob import glob
import numpy as np

import json

import random

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=100,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=2.2,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=2.2,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=400, #800!
                    help='Resolution of the images.')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
parser.add_argument('--name', type=str, default='model',
                    help='Model id')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth # ('8', '16')
render.image_settings.file_format = args.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = True

#set up nodes to change background color

bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
composite = tree.nodes[0]
render_layers = tree.nodes[1]
alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
links = tree.links
link_1 = links.new(render_layers.outputs[0], alpha_over.inputs[2])
link_2 = links.new(alpha_over.outputs[0], composite.inputs[0])
alpha_over.inputs[1].default_value = (1, 1, 1, 0)


scene.use_nodes = True

scene.view_layers["ViewLayer"].use_pass_normal = True
scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
scene.view_layers["ViewLayer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

bpy.context.scene.render.use_persistent_data = True

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = ''
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = args.format
depth_file_output.format.color_depth = args.color_depth
if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
    depth_file_output.format.color_mode = "BW"

    # Remap as other types can not represent the full range of depth.
    map = nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.offset = [-0.7]
    map.size = [args.depth_scale]
    map.use_min = True
    map.min = [0]

    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

# Create normal output nodes
scale_node = nodes.new(type="CompositorNodeMixRGB")
scale_node.blend_type = 'MULTIPLY'
# scale_node.use_alpha = True
scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

bias_node = nodes.new(type="CompositorNodeMixRGB")
bias_node.blend_type = 'ADD'
# bias_node.use_alpha = True
bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_node.outputs[0], bias_node.inputs[1])

normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
normal_file_output.base_path = ''
normal_file_output.file_slots[0].use_node_format = True
normal_file_output.format.file_format = args.format
links.new(bias_node.outputs[0], normal_file_output.inputs[0])

# Create albedo output nodes
alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
albedo_file_output.base_path = ''
albedo_file_output.file_slots[0].use_node_format = True
albedo_file_output.format.file_format = args.format
albedo_file_output.format.color_mode = 'RGBA'
albedo_file_output.format.color_depth = args.color_depth
links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = 'ID Output'
id_file_output.base_path = ''
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = args.format
id_file_output.format.color_depth = args.color_depth

if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
else:
    id_file_output.format.color_mode = 'BW'

    divide_node = nodes.new(type='CompositorNodeMath')
    divide_node.operation = 'DIVIDE'
    divide_node.use_clamp = False
    divide_node.inputs[1].default_value = 2**int(args.color_depth)

    links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
    links.new(divide_node.outputs[0], id_file_output.inputs[0])

# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')

bpy.ops.import_scene.gltf(filepath=args.obj)


obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

# create material
#mat = bpy.data.materials.new(name='Material')

#obj.data.materials.append(mat)
#mat.use_nodes=True

# let's create a variable to store our list of nodes
#mat_nodes = mat.node_tree.nodes

# let's set the metallic to 1.0
#mat_nodes['Principled BSDF'].inputs['Metallic'].default_value=1.0
#mat_nodes['Principled BSDF'].inputs['Roughness'].default_value=0.0

# Possibly disable specular shading
for slot in obj.material_slots:
    node = slot.material.node_tree.nodes['Principled BSDF']
    node.inputs['Specular'].default_value = 0.3
    node.inputs['Metallic'].default_value=0.5
    node.inputs['Roughness'].default_value=0.25

if args.scale != 1:
    bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
    bpy.ops.object.transform_apply(scale=True)
#if args.remove_doubles:
#    bpy.ops.object.mode_set(mode='EDIT')
#    bpy.ops.mesh.remove_doubles()
#    bpy.ops.object.mode_set(mode='OBJECT')
#if args.edge_split:
#    bpy.ops.object.modifier_add(type='EDGE_SPLIT')
#    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
#    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

# Set objekt IDs
obj.pass_index = 1

 #Make light just directional, disable shadows.
light = bpy.data.lights['Light']
light.type = 'SUN'
light.use_shadow = True
 #Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 0.0

# create light datablock, set attributes
light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
light_data.energy = 200
light_data.specular_factor = 0.4
light_data.use_shadow = True
#light_data.color = (1.0,0,0)

# create new object with our light datablock
light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

# link light object
bpy.context.collection.objects.link(light_object)

# make it active 
bpy.context.view_layer.objects.active = light_object

#change location
light_object.location = (4, 1, 1)

# Add another light source so stuff facing away from light is not completely dark
#bpy.ops.object.light_add(type='SUN')
#light2 = bpy.data.lights['Sun']
#light2.use_shadow = True
#light2.specular_factor = 1.0
#light2.energy = 0.045
#bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
#bpy.data.objects['Sun'].rotation_euler[0] += 180

# Place camera
cam = scene.objects['Camera']
cam.location = (0, 3.2, 0)
cam.data.angle_x = 0.6911112070083618
#cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

model_identifier = args.name
fp = os.path.join(os.path.abspath(args.output_folder), model_identifier)


for i in range(0, args.views):

    rot = np.random.uniform(0.001, 1, size=3) * (1,0,2*np.pi) # ( gora-dol, , z prawo-lewo)
    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
    cam_empty.rotation_euler = rot

    render_file_path = fp + f'/image_{i}'

    scene.render.filepath = render_file_path

    bpy.ops.render.render(write_still=True)  # render still

    with open(fp+f"/image_{i}.json", "w") as file:
        json.dump(listify_matrix(cam.matrix_world), file, indent=4)

