import cv2
import os
import datetime
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, \
    OperationStatusType
import pandas as pd
from time import sleep

# azure api에 접근하기 위한 설정 및 특정인 사진 입력
KEY = '295503c2dff440a097ba328d6a56c950'
ENDPOINT = 'https://nakg.cognitiveservices.azure.com/'
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

person_list = pd.read_csv('./data/personlist.csv')

name = 0


def logging(persisted_face_id):
    target = person_list.loc[person_list['face_id'] == persisted_face_id]  # 한개만 있는게 보장
    target_person = target.loc[target['check'] == True]
    df2 = person_list.loc[person_list['face_id'] == persisted_face_id]

    if (target_person.empty == False):
        log_file = pd.read_csv('./data/log_file.csv')
        temp_target = target_person.copy(deep=True)
        time = datetime.datetime.now()
        now = time.strftime('%Y-%m-%d %H-%M-%S')
        temp_target["Time"] = now
        temp_determine_time = log_file.loc[log_file['face_id'] == persisted_face_id]
        temp_determine_time = temp_determine_time[:100]
        temp_log2 = temp_determine_time.loc[temp_determine_time["Time"] == now]
        # temp_target['Reco_path'] = record_ref()
        if (temp_log2.empty == True):  # 같은시간에 기록 안된경우
            log_file = log_file.append(temp_target, ignore_index=True)
            log_file.to_csv("./data/log_file.csv", index=False)

        elif (temp_determine_time.empty == True):  # 아예 기록 안되어있는경우
            log_file = log_file.append(temp_target, ignore_index=True)
            log_file.to_csv("./data/log_file.csv", index=False)
    #print( df2)
    return df2['name'].values[0]


while True:
    try:
        # 로드되지 않은 이미지 거름(마지막 이미지)
        discard = open("../picture/" + str(name) + ".jpg", 'rb')
        sleep(0.01)
        frame = open("../picture/" + str(name) + ".jpg", 'rb')
        print(name)
        detected_face = face_client.face.detect_with_stream(image=frame, return_face_id=True,
                                                            recognition_model='recognition_03')
        flag = 0
        confidence = 0
        person_name = ""
        if detected_face:
            print('Detected face ID fr-om', str(name), ':')
            for face in detected_face:
                print("발견" + face.face_id)
                similar_face = face_client.face.find_similar(face_id=face.face_id,
                                                             face_list_id='test')
                # 의미없는 반복문이긴 하나, SIMILAR_FACE가 없는 경우를 위해서 만듬.
                for i in similar_face:
                    # log 기록하는 csv 파일 열기 , list 포함하고있는 파일도 열기
                    if similar_face[0].confidence > 0.6:
                        print(similar_face[0].persisted_face_id)
                        print("yes", name)
                        flag = 1
                        confidence = similar_face[0].confidence
                        # 사람 정보 반환
                        person_name = logging(similar_face[0].persisted_face_id)
                        f = open("../atoy/name.txt", 'a')
                        data = person_name + "\n"
                        f.write(data)
                        f.close()
                        # 정확도 반환
                        f = open("../atoy/confidence.txt", 'a')
                        data = str(confidence) + "\n"
                        f.write(data)
                        f.close()
                        # 탐색결과 저장
                        f = open("../atoy/result.txt", 'a')
                        data = str(name) + "\n"
                        f.write(data)
                        f.close()

        name += 1
    except FileNotFoundError:
        # sleep(0.3)
        continue



"""
while True:
    try:
        frame_1 = cv2.imread("../picture/" + str(name + 1) + ".jpg")
        if not frame_1.any():
            continue
        frame = cv2.imread("../picture/" + str(name) + ".jpg")
        cv2.imshow("frame", frame)
        if cv2.waitKey(33) > 0:
            break
        sleep(0.3)
        f = open("../result.txt", 'a')
        data = str(name) + "\n"
        f.write(data)
        f.close()
        name += 1
    except:
        if cv2.waitKey(33) > 0:
            break
        continue


def API_CALL(name, face_client):
    path = "./tmp/"
    multi_face_image_path = path + name
    multi_image_name = os.path.basename(multi_face_image_path)
    image_name_2 = open(multi_face_image_path, 'rb')
    detected_faces2 = face_client.face.detect_with_stream(image=image_name_2, return_face_id=True,
                                                          recognition_model='recognition_03')
    flag = 0
    confidence = 0
    person_name = ""
    if detected_faces2:
        print('Detected face ID from', multi_image_name, ':')
        for face in detected_faces2:
            print("발견" + face.face_id)
            similar_faces = face_client.face.find_similar(face_id=face.face_id,
                                                          face_list_id='api_list')
            # 의미없는 반복문이긴 하나, SIMILAR_FACE가 없는 경우를 위해서 만듬.
            for i in similar_faces:
                # log 기록하는 csv 파일 열기 , list 포함하고있는 파일도 열기
                if similar_faces[0].confidence > 0.6:
                    flag = 1
                    confidence = similar_faces[0].confidence

    return flag, confidence, person_name
"""
