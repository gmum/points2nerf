import os
import bpy
import numpy as np
from random import randrange
import glob
import traceback
import argparse

def get_materials_info(material_slots):
    images = []
    for material_slot in material_slots:
        image = None
        color = None
        local_pixels = None
        for node in material_slot.material.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                image = bpy.data.images[node.image.name]
                local_pixels = list(image.pixels[:])
            elif node.type == 'BSDF_PRINCIPLED':
                color = node.color
        if image != None:
            images.append(('img', image, local_pixels))
        else:
            images.append(('col', color))
    return images

def clamp_uv(val):
    return max(0, min(val, 1))

def should_skip(selected_verts, vert_idx, inserted, ob):
    try:
        index = selected_verts.index(ob.data.vertices[vert_idx])
        if inserted[index] == True:
            return True
        inserted[index] = True
        return False
    except ValueError:
        return True

def append_vert_and_color(verts_coordinates, verts_colors, ob, vert_idx, loop_idx, face, images, width, height, local_pixels):
    if images[face.material_index][0] == 'col':
        verts_coordinates.append((ob.data.vertices[vert_idx].co[0], ob.data.vertices[vert_idx].co[1], ob.data.vertices[vert_idx].co[2]))
        verts_colors.append((images[face.material_index][1][0], images[face.material_index][1][1], images[face.material_index][1][2], 1))
    else:
        uv_coords = ob.data.uv_layers.active.data[loop_idx].uv
        
        target = [round(clamp_uv(uv_coords.x) * (width - 1)), round(clamp_uv(uv_coords.y) * (height - 1))]
        index = ( target[1] * width + target[0] ) * 4
    
        verts_coordinates.append((ob.data.vertices[vert_idx].co[0], ob.data.vertices[vert_idx].co[1], ob.data.vertices[vert_idx].co[2]))
        verts_colors.append((local_pixels[index], local_pixels[index + 1], local_pixels[index + 2], 1))

def process_faces_for_cloud(ob, images, inserted):
    verts_coordinates, verts_colors = [], []
    image, local_pixels, width, height = None, None, None, None
    for face in ob.data.polygons:
        if images[face.material_index][0] == 'img':
            image = images[face.material_index][1]
            local_pixels = images[face.material_index][2]
            width = image.size[0]
            height = image.size[1]
        else:
            image = images[face.material_index][1][2]
            
        for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
            if should_skip(selected_verts, vert_idx, inserted, ob):
                continue
            append_vert_and_color(verts_coordinates, verts_colors, ob, vert_idx, loop_idx, face, images, width, height, local_pixels)
    return verts_coordinates, verts_colors

def export_to_cloud(ob, filepath : str, selected_verts):
    images = get_materials_info(ob.material_slots)
    inserted = np.zeros((len(selected_verts)), dtype=bool)
    
    verts_coordinates, verts_colors = process_faces_for_cloud(ob, images, inserted)        
    
    np.savetxt(filepath + ob.name + '_mesh_data.txt', np.asarray(verts_coordinates), delimiter=' ', fmt='%f')
    np.savetxt(filepath + ob.name + '_color_data.txt', np.asarray(verts_colors), delimiter=' ', fmt='%f')

def bake(filepath : str, selected_verts):
    bpy.context.view_layer.objects.active = bpy.data.objects[0]
    bpy.ops.object.mode_set(mode='VERTEX_PAINT')

    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, 'vertices') == False:
            continue
        
        export_to_cloud(obj, filepath, selected_verts)
    bpy.ops.object.mode_set(mode='OBJECT')


def get_axis_index(pos, bound_lower, bound_upper, interval_len, axis_len):
    x = pos
    x_start = bound_lower + interval_len
    x_index = 0
    while(x_start < x):
        x_start += interval_len
        x_index += 1
    if x_index >= axis_len:
        x_index = axis_len - 1
    return x_index
    

def get_box_index(vert, bounds, interval_len, axis_len):
    return (get_axis_index(vert.co[0], bounds[0][0], bounds[0][1], interval_len[0], axis_len), get_axis_index(vert.co[1], bounds[1][0], bounds[1][1], interval_len[1], axis_len), get_axis_index(vert.co[2], bounds[2][0], bounds[2][1], interval_len[2], axis_len))
        

def get_random_vertices(boxes, axis_len):
    selected_verts = []

    while(len(selected_verts) < axis_len**3):
        box_index = randrange(len(boxes))
        box_len = len(boxes[box_index])
        vert = boxes[box_index].pop(randrange(box_len))
        if vert in selected_verts:
            continue
        selected_verts.append(vert)
        if len(boxes[box_index]) == 0:
            boxes.pop(box_index)
    return selected_verts

def put_verts_to_boxes(boxes, vertices, bounds, x_interval_len, y_interval_len, z_interval_len, axis_len):
    for vert in vertices:
        box_index = get_box_index(vert, bounds, (x_interval_len, y_interval_len, z_interval_len), axis_len)
        if boxes[box_index[0]][box_index[1]][box_index[2]] == None:
            boxes[box_index[0]][box_index[1]][box_index[2]] = []
        boxes[box_index[0]][box_index[1]][box_index[2]].append(vert)

def select_verts_subspace(obj, axis_len):
    x_sort = sorted(obj.data.vertices, key=lambda v: v.co[0])
    y_sort = sorted(obj.data.vertices, key=lambda v: v.co[1])
    z_sort = sorted(obj.data.vertices, key=lambda v: v.co[2])
    
    bounds = ((x_sort[0].co[0], x_sort[-1].co[0]), (y_sort[0].co[1], y_sort[-1].co[1]), (z_sort[0].co[2], z_sort[-1].co[2]))
    
    x_interval_len = (bounds[0][1] - bounds[0][0]) / axis_len
    y_interval_len = (bounds[1][1] - bounds[1][0]) / axis_len
    z_interval_len = (bounds[2][1] - bounds[2][0]) / axis_len
    
    boxes = np.empty((axis_len, axis_len, axis_len), dtype=type(list))

    put_verts_to_boxes(boxes, x_sort, bounds, x_interval_len, y_interval_len, z_interval_len, axis_len)

    boxes = boxes.flatten()
    boxes = [box for box in boxes if box != None]
    
    return get_random_vertices(boxes, axis_len)


def get_subdiv_amount(vert_len, axis_len):
    subdiv_threshold = 4**2 * axis_len**3
    subdiv_amount = 0
    vert_len *= 4
    while vert_len < subdiv_threshold:
        subdiv_amount += 1
        vert_len *= 4
    return subdiv_amount

def merge_doubles(merge_threshold = 0.0000001):
    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, 'vertices') == False:
            continue
        print('before doubles merge: ', len(obj.data.vertices))
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold = merge_threshold)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type = 'FACE')
        bpy.ops.mesh.select_interior_faces()
        bpy.ops.mesh.delete(type='FACE')
        bpy.ops.object.mode_set(mode='OBJECT')
        print('after doubles merge: ', len(obj.data.vertices))
    

def apply_subsurf(axis_len):
    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, 'vertices') == False:
            continue
        subdiv_amount = 1
        if subdiv_amount == 0:
            return
        
        bpy.context.view_layer.objects.active = obj
        while(len(obj.data.vertices) < axis_len**3):
            bpy.ops.object.modifier_add(type='SUBSURF')
            bpy.context.object.modifiers[0].subdivision_type = 'SIMPLE'
            bpy.context.object.modifiers[0].levels = subdiv_amount
            bpy.ops.object.modifier_apply(modifier='Subdivision')

def apply_simple_subdivide(min_vert_count):
    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, 'vertices') == False:
            continue
        bpy.context.view_layer.objects.active = obj
        while(len(obj.data.vertices) < min_vert_count):
#            print('subdiv', len(obj.data.vertices))  
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all()
            bpy.ops.mesh.subdivide(number_cuts=1)
            bpy.ops.object.mode_set(mode="OBJECT")
        

def cleanup_mesh():
    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, 'vertices') == False:
            continue
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode="OBJECT")

def apply_decimate():
    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, 'vertices') == False:
            continue
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='DECIMATE')
        bpy.context.object.modifiers[0].ratio = 0.5
        bpy.ops.object.modifier_apply(modifier='Decimate')


def get_models_directories(path):
    obj_list = []
    dir_list = sorted(os.listdir(path))
    dir_list = [os.path.join(path, dir) for dir in dir_list if os.path.isdir(os.path.join(path, dir))]
    for dir in dir_list:
        for r, d, files in os.walk(dir):
            if 'images' not in d:
                continue
            for r1, d1, f1 in os.walk(os.path.join(dir, 'models')):
                for file in f1:
                    if file.endswith('.obj'):
                        obj_list.append(os.path.join(dir, 'models', file))
    
    print('models with textures: ', len(obj_list))
    return obj_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export cloud of points')
    parser.add_argument('shapenet_path', type=str,
                        help='Relative shapenet path (./shapenet/02958343)')
    parser.add_argument('export_path', type=str,
                        help='Relative output path (./dataset_maker/export/02958343)')
    parser.add_argument('check_path', type=str,
                        help='Relative previous run output path (./dataset_maker/data/02958343/*.txt) to continue sampling in case of error')
                        
    args = parser.parse_args()

    path_to_obj_dir = args.shapenet_path 
    export_path = args.export_path
    check_path = args.check_path 
    
    axis_len = 3
    min_vert_subdiv_count = 10000
    
    bpy.ops.object.delete({"selected_objects": bpy.context.scene.objects})
    
    already_generated = glob.glob(check_path)
    already_generated_names = [x.split('\\')[-1].split('_')[0] for x in already_generated]
    print(already_generated_names)
    
    obj_list = get_models_directories(path_to_obj_dir)
    i = 0
    for file in obj_list:
        try:
            print('file ', file)
            name = file.split('\\')[-3]
            if name in already_generated_names:
                print(name, ' is already generated.')
                continue
            bpy.ops.import_scene.obj(filepath = file, use_split_objects=False)
            for obj in bpy.context.scene.objects:
                obj.name = name
                print('set name ', obj.name)
            cleanup_mesh()
            apply_decimate()
            apply_simple_subdivide(min_vert_subdiv_count)
            selected_verts = select_verts_subspace(bpy.data.objects[0], axis_len)
            print('selected verts len ', len(selected_verts))
            bake(export_path, selected_verts)
            bpy.context.view_layer.objects.active = bpy.data.objects[0]
            bpy.ops.object.delete({"selected_objects": bpy.context.scene.objects})
            i += 1
            print('export progress: ', (i * 100) / len(obj_list), ' %')
        except Exception as e:
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.delete({"selected_objects": bpy.context.scene.objects})
            f=open("cloud_error.txt", "a")
            f.write(file + '\n')
            f.write(str(e) + '\n')
            f.write(traceback.format_exc())
            f.close()
            print(e)
    print("Shapenet size: ", len(obj_list))
    print('Generated :', i)
    print('Already there: ', len(already_generated_names))