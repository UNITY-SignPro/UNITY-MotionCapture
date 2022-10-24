import bpy
import json
from math import *

# bpy 초기화
# clear scene

for elem in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[elem.name], do_unlink=True)
# fbx 메인 모델 불러오기

# bpy.ops.mesh.primitive_plane_add(location=(0.0, 0.0, 0.0))

bpy.ops.import_scene.fbx(
    filepath="default2.fbx"
)

ob = bpy.data.objects['Armature']
bpy.ops.object.mode_set(mode='POSE')


# bpy.context.scene.frame_set(0)
# Man = bpy.data.objects['Armature']

# Man.location = (0, 0, -1.5)
# Man.keyframe_insert(data_path="location", index=-1)


# Func 관절 기초값 설정
def defaultSetting(part, facing, angle):
    pbone = ob.pose.bones[part]
    pbone.rotation_mode = 'XYZ'
    pbone.rotation_euler.rotate_axis(facing, angle)
# Func 윗팔 각도 조정
def UpperArmX(part, rotation, frame):
    if "Left" in part:
        pbone = ob.pose.bones[part]
        pbone.rotation_mode = 'XYZ'
        pbone.rotation_euler.rotate_axis("Y", -rotation[1])
        pbone.rotation_euler.rotate_axis("X", rotation[0])
        pbone.keyframe_insert(data_path="rotation_euler", frame = frame)
        pbone.rotation_euler.rotate_axis("X", -rotation[0])
        pbone.rotation_euler.rotate_axis("Y", rotation[1])
    else:
        pbone = ob.pose.bones[part]
        pbone.rotation_mode = 'XYZ'
        pbone.rotation_euler.rotate_axis("Y", rotation[1])
        pbone.rotation_euler.rotate_axis("X", rotation[0])
        pbone.keyframe_insert(data_path="rotation_euler", frame = frame)
        pbone.rotation_euler.rotate_axis("X", -rotation[0])
        pbone.rotation_euler.rotate_axis("Y", -rotation[1])



# pbone = ob.pose.bones["LeftLeg"]
# pbone.rotation_mode = 'XYZ'
# pbone.rotation_euler.rotate_axis("X", radians(-90))
# pbone = ob.pose.bones["RightLeg"]
# pbone.rotation_mode = 'XYZ'
# pbone.rotation_euler.rotate_axis("X", radians(-90))
# Func 윗팔 각도 조정
pbone = ob.pose.bones["LeftForeArm"]
pbone.rotation_mode = 'XYZ'
pbone.rotation_euler.rotate_axis("Y", radians(-60))
pbone = ob.pose.bones["RightForeArm"]
pbone.rotation_mode = 'XYZ'
pbone.rotation_euler.rotate_axis("Y", radians(60))



def LowerArm(part, rotation, frame):
    pbone = ob.pose.bones[part]
    pbone.rotation_mode = 'XYZ'
    pbone.rotation_euler.rotate_axis("X", rotation)
    pbone.keyframe_insert(data_path="rotation_euler", frame = frame)
    pbone.rotation_euler.rotate_axis("X", -rotation)

# Func 손가락 각도 조정
def fingerRotate(part, rotation, frame):
    pbone = ob.pose.bones[part]
    pbone.rotation_mode = 'XYZ'
    if "Thumb" in part:
        weight = 1
        if "Left" in part:
            weight = -1
        pbone.rotation_euler.rotate_axis("Z", -rotation * weight)
        pbone.keyframe_insert(data_path="rotation_euler", frame = frame)
        pbone.rotation_euler.rotate_axis("Z", rotation * weight)
    else:
        pbone.rotation_euler.rotate_axis("X", -rotation)
        pbone.keyframe_insert(data_path="rotation_euler", frame = frame)
        pbone.rotation_euler.rotate_axis("X", rotation)

# Func 손목 각도 조정
def Snap_Rotation(part, rotation, frame):
    pbone = ob.pose.bones[part]
    pbone.rotation_mode = 'XYZ'

    # is right?
    if "Left" not in part:
        rotation *= -1
    # print(rotation)
    # rotation = 3.14159
    rotation = radians(rotation)
    pbone.rotation_euler.rotate_axis("Y", rotation)
    pbone.keyframe_insert(data_path="rotation_euler", frame = frame)
    pbone.rotation_euler.rotate_axis("Y", -rotation)

# json을 통한 모델 리깅
def Save_FBX(labels):
    pre_frame = 0
    for label in labels:
        with open(f'./json/{label}.json', 'r') as f:
            frames = json.load(f)
        last_frame = int([*frames.keys()][-1].replace("frame_", ""))
        for fps in range(last_frame):
            # 어깨 - 팔 각도 조정
            # print(frames[f"frame_{fps}"]["left"]["upperArm"][0])
            UpperArmX("LeftArm", frames[f"frame_{fps}"]["left"]["upperArm"], pre_frame + fps)
            UpperArmX("RightArm", frames[f"frame_{fps}"]["right"]["upperArm"], pre_frame + fps)
            #
            LowerArm("LeftForeArm", frames[f"frame_{fps}"]["left"]["lowerArm"][0], pre_frame + fps)
            LowerArm("RightForeArm", frames[f"frame_{fps}"]["right"]["lowerArm"][0], pre_frame + fps)

            # 손가락 각도 조정
            for name in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                for idx in range(1, 4):
                    if "hand" in frames[f"frame_{fps}"]["left"]:
                        fingerRotate(f"LeftHand{name}{idx}", -frames[f"frame_{fps}"]["left"]["hand"][name][str(idx)], pre_frame + fps)
                        Snap_Rotation("LeftHand", frames[f"frame_{fps}"]["left"]["hand"]["facing"], pre_frame + fps)
                    if "hand" in frames[f"frame_{fps}"]["right"]:
                        fingerRotate(f"RightHand{name}{idx}", -frames[f"frame_{fps}"]["right"]["hand"][name][str(idx)], pre_frame + fps)
                        Snap_Rotation("RightHand", frames[f"frame_{fps}"]["right"]["hand"]["facing"], pre_frame + fps)

        pre_frame = fps + 1

    bpy.data.actions[0].name = "anime"
    # # #fbx 저장
    # bpy.ops.export_scene.fbx(
    #     filepath=bpy.path.abspath("test.fbx"),
    #     use_active_collection=True,
    # )

    # gltf 저장
    bpy.ops.export_scene.gltf(
        filepath=bpy.path.abspath("test.gltf"),
        # use_active_collection = True,
        export_format = 'GLTF_EMBEDDED',
    )
    print(f"{label} is done")

Save_FBX(['(활을 쏘는)사수', '(존재가)없다'])