import cv2
import json
from math import *
import mediapipe as mp
import numpy as np
import time

# ===============================
#         포즈 데이터 생성
# ===============================

# 손가락 관련 함수
def toVec(head, tail):
    return [ tail.x - head.x, tail.y - head.y, tail.z - head.z ]
def vec2Rotate(head, tail):
    vecIn = head[0]*tail[0] + head[1]*tail[1] + head[2]*tail[2]
    if vecIn == 0:
        return 0
    vec01 = sqrt(pow(head[0], 2) + pow(head[1], 2) + pow(head[2], 2))
    vec02 = sqrt(pow(tail[0], 2) + pow(tail[1], 2) + pow(tail[2], 2))
    return acos(vecIn/(vec01*vec02))
def fingerRotation(hand):
    fingers = [
        [[0, 1], [1, 2], [2, 3], [3, 4]],        # 엄지 손가락
        [[0, 5], [5, 6], [6, 7], [7, 8]],        # 중지 손가락
        [[0, 9], [9, 10], [10, 11], [11, 12]],   # 검지 손가락
        [[0, 13], [13, 14], [14, 15], [15, 16]], # 약지 손가락
        [[0, 17], [17, 18], [18, 19], [19, 20]]  # 소지 손가락
    ]
    fingerLines = []
    joints = []
    for finger in fingers:
        temp = []
        for fl in finger:
            temp.append(toVec(hand.landmark[fl[0]], hand.landmark[fl[1]]))
        fingerLines.append(temp)
    for fl in fingerLines:
        for joint in range(3):
            joints.append(vec2Rotate(fl[joint], fl[joint + 1]))
    return joints
def Hand_Thumb_Rel(landmarks):
    __hor__ = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value].x - landmarks[mp_hands.HandLandmark.RING_FINGER_MCP.value].x
    __ver__ = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value].y - landmarks[mp_hands.HandLandmark.RING_FINGER_MCP.value].y
    hor = 0 if __hor__ < 0 else 180
    ver = 90 if __ver__ < 0 else 270
    # print(__hor__, __ver__)
    # 상 하 좌 우
    if abs(__ver__)/abs(__hor__) >= 1:
        return ver
    return hor

# 허리 - 어깨 - 팔굼치까지의 3점을 기반으로 팔 높이 각도 계산
def ShoulderUpperAngle_Y(landmarks, type):
    if type == "LEFT":
        hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
    else:
        hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x ,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
    radians = np.arctan2(elbow[1]-shoulder[1], elbow[0]-shoulder[0]) - np.arctan2(hip[1]-shoulder[1], hip[0]-shoulder[0])
    angle = np.abs(degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return np.radians(angle)
# 반대 어꺠 - 어꺠 - 팔꿈치 3점을 기반으로 팔이 떨어진 각도 계산
def ShoulderUpperAngle_X(landmarks, type):
    if type == "LEFT":
        Ashoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
    else:
        Ashoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
    radians = np.arctan2(elbow[1]-shoulder[1], elbow[0]-shoulder[0]) - np.arctan2(Ashoulder[1]-shoulder[1], Ashoulder[0]-shoulder[0])
    angle = np.abs(degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    angle -= 90
    return np.radians(90 - angle)
# 팔이 굽혀진 걱도 계산
def ShoulderLowerAngle(landmarks, type):
    if type == "LEFT":
        shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
        wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
    else:
        shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
        wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
    radians = np.arctan2(wrist[1]-elbow[1], wrist[0]-elbow[0]) - np.arctan2(shoulder[1]-elbow[1], shoulder[0]-elbow[0])
    angle = np.abs(degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    # 가중치 140 ( 최대 140 )
    return np.radians( (180-angle)/180*140 )

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
# 동영상 기반 프레임 뽑아냄
def returnPoseDictionary(video):
    frames = {}
    fps = 0; cap = cv2.VideoCapture(video)
    leftShoulder = 0; rightShoulder = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            results = pose.process(image)
            frame = {}
            if results.pose_landmarks:
                if not leftShoulder:
                    leftShoulder = (results.pose_landmarks.landmark[11])
                if not rightShoulder:
                    rightShoulder = (results.pose_landmarks.landmark[12])
                landmarks = results.pose_landmarks.landmark
                frame["left"] = {"upperArm" : [ShoulderUpperAngle_X(landmarks, "LEFT"),
                                               ShoulderUpperAngle_Y(landmarks, "LEFT")],
                                 "lowerArm" : [ShoulderLowerAngle(landmarks, "LEFT")]}
                frame["right"] = {"upperArm" : [ShoulderUpperAngle_X(landmarks, "RIGHT"),
                                               ShoulderUpperAngle_Y(landmarks, "RIGHT")],
                                 "lowerArm" : [ShoulderLowerAngle(landmarks, "RIGHT")]}
                frames[f"frame_{fps}"] = frame
                fps += 1

    # 손가락
    fps = 0; cap = cv2.VideoCapture(video)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            fps += 1
            success, image = cap.read()
            if not success: break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    hand_facing = results.multi_handedness[num].classification[0].label
                    hands_data = fingerRotation(hand)

                    frame = {
                        "Thumb" : {
                            "1" : hands_data[0],
                            "2" : hands_data[1],
                            "3" : hands_data[2],
                        },
                        "Index" : {
                            "1" : hands_data[3],
                            "2" : hands_data[4],
                            "3" : hands_data[5],
                        },
                        "Middle" : {
                            "1" : hands_data[6],
                            "2" : hands_data[7],
                            "3" : hands_data[8],
                        },
                        "Ring" : {
                            "1" : hands_data[9],
                            "2" : hands_data[10],
                            "3" : hands_data[11],
                        },
                        "Pinky" : {
                            "1" : hands_data[12],
                            "2" : hands_data[13],
                            "3" : hands_data[14],
                        }
                    }
                    if hand_facing.lower() == "left":
                        frames[f"frame_{fps}"]["right"]["hand"] = frame

                        frames[f"frame_{fps}"]["right"]["hand"]["facing"] = Hand_Thumb_Rel(results.multi_hand_landmarks[num].landmark)
                    else:
                        frames[f"frame_{fps}"]["left"]["hand"] = frame
                        frames[f"frame_{fps}"]["left"]["hand"]["facing"] = Hand_Thumb_Rel(results.multi_hand_landmarks[num].landmark)
    cap.release()
    cv2.destroyAllWindows()
    return frames
# 프레임에서 손가락 데이터 보정
def correctionFingerRotation(frames):
    last_frame = int([*frames.keys()][-1].replace("frame_", ""))
    __lastStep__ = last_frame // 5 + (0 if last_frame%5 == 0 else 1)
    for __frame__ in range(__lastStep__):
        if __frame__ * 5 <= last_frame:
            if "hand" in frames[f"frame_{__frame__ * 5}"]["left"]:
                for __finger__ in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                    # 1차 함수 생성
                    for __joint__ in ["1", "2", "3"]:
                        fr = frames[f"frame_{__frame__ * 5}"]["left"]["hand"][__finger__][__joint__]
                        # 만약 마지막 프레임에 손이 없으면...
                        for __holy__ in range(4, -1, -1):
                            if "hand" in frames[f"frame_{__frame__ * 5 + __holy__}"]["left"]:
                                ls = frames[f"frame_{__frame__ * 5 + __holy__}"]["left"]["hand"][__finger__][__joint__]
                                break
                        m = (ls - fr) / 4
                        b = fr
                        for __relateve__ in range(5):
                            __rFrame__ = __frame__ * 5 + __relateve__
                            if "hand" in frames[f"frame_{__rFrame__}"]["left"]:
                                frames[f"frame_{__rFrame__ }"]["left"]["hand"][__finger__][__joint__] = m * __relateve__ + b
    return frames
# json파일로 프레임 데이터 저장
def savePoseJson(video, label):
    startTime = time.time()
    frames = returnPoseDictionary(video)
    frames = correctionFingerRotation(frames)
    with open(f'{label}.json', 'w', encoding='utf-8') as make_file:
        json.dump(frames, make_file, indent="\t")
    print(f'create {label}.json pose file     Working Time : {time.time() - startTime:.3f} sec')


# savePoseJson("https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20191028/631984/MOV000248548_700X466.webm", "./json/yee")


