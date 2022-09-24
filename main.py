import bpy
# 수학
import math
import mathutils

# clear scene
for elem in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[elem.name], do_unlink=True)

# fbx 불러오기
bpy.ops.import_scene.fbx(
    filepath="man.fbx"
)



# armature = bpy.data.objects["Armature"]
#
# for bone in armature.pose.bones:
#     print(bone)



ob = bpy.data.objects['Armature']
bpy.ops.object.mode_set(mode='POSE')

pbone = ob.pose.bones["root"]

pbone.rotation_mode = 'XYZ'
axis = 'Z'
angle = 120
for angle in range(100):
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))
    bpy.ops.object.mode_set(mode='OBJECT')
    pbone.keyframe_insert(data_path="rotation_euler" , frame = angle)



#fbx 저장
bpy.ops.export_scene.fbx(
    filepath=bpy.path.abspath("test.fbx"),
    use_active_collection=True
)
