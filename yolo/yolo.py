import cv2
import numpy as np
from time import sleep
import time


def color_detection(cloth):
    size = 20
    picture = cloth

    white = 0
    black = 0
    red = 0  # (166 ~ 180, 0 ~ 15)
    yellow = 0  # (16 ~ 45)
    green = 0  # (46 ~ 75)
    blue = 0  # (76 ~ 135)
    purple = 0  # (136 ~ 165)
    background = 0

    src = picture
    src = cv2.resize(src, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    height, width, NOT_USED = hsv.shape

    for i in range(0, height):
        for j in range(0, width):
            if (s[i][j] == 0) and (v[i][j] == 0):
                background += 1

            elif 0 < s[i][j] <= 50:
                if 0 < v[i][j] <= 180:
                    black += 1
                elif 190 <= v[i][j] <= 255:
                    white += 1

            else:
                if (0 <= h[i][j] <= 13) or (166 <= h[i][j] <= 180):
                    red += 1
                elif 11 <= h[i][j] <= 35:
                    yellow += 1
                elif 36 <= h[i][j] <= 84:
                    green += 1
                elif 85 <= h[i][j] <= 125:
                    blue += 1
                elif 126 <= h[i][j] <= 165:
                    purple += 1

    # print(black, white, red, yellow, green, blue, purple, background)

    color2 = [black, white, red, yellow, green, blue, purple]
    colortext = {0: 'black', 1: 'white', 2: 'red', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'purple'}
    # 리스트 참조가 아닌 복사
    tmp2 = color2[:]

    tmp2.sort(reverse=True)

    return colortext[color2.index(tmp2[0])], colortext[color2.index(tmp2[1])]


def mainfunction1(video_1, whatcolor1=None):

    # ==========================================================
    # yolo 알고리즘 사용 위한 weights, cfg, name 파일
    net_1 = cv2.dnn.readNet("./data/weights/yolov3-wider_16000.weights", "./data/cfg/yolov3-face.cfg")
    net_2 = cv2.dnn.readNet("./data/weights/yolov3-wider_16000.weights", "./data/cfg/yolov3-face.cfg")

    with open("./data/names/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names_1 = net_1.getLayerNames()
    output_layers_1 = [layer_names_1[i[0] - 1] for i in net_1.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # GPU 가속
    net_1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # ==========================================================
    # 탐색 객체 좌표 기록 위한 변수
    pixelgap = 40
    resetframe = 50
    xylist_1 = []
    truelist_1 = []
    trueinform_1 = []


    # 이미지 이름 변경 위한 count 변수
    count_1 = 0


    # 영상 입력 객체 생성
    cap_1 = cv2.VideoCapture(video_1)

    # color detection 를 위한 변수 및 모듈
    if whatcolor1 is None:
        whatcolor1 = []
    rl = np.array([0, 0, 190])
    rh = np.array([180, 70, 255])
    # fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=0)

    # ==========================================================
    while True:

        ret_1, frame_1 = cap_1.read()
        # 현재 프레임 == 총 프레임 : 비디오의 끝 종료
        if cap_1.get(cv2.CAP_PROP_POS_FRAMES) == cap_1.get(cv2.CAP_PROP_FRAME_COUNT):
            print("video 1 end")
            break

        # 프레임 수 조절 (캠 사용시 사용 X)2
        if cap_1.get(cv2.CAP_PROP_POS_FRAMES) % 1 != 0:
            continue

        # 설정 값마다 좌표 리스트 삭제
        if cap_1.get(cv2.CAP_PROP_POS_FRAMES) % resetframe == 0:
            del xylist_1[:]
            del truelist_1[:]
            del trueinform_1[:]

        # 영상 회전 (필요시 설정)
        if 0:
            height_1, width_1, channel_1 = frame_1.shape
            matrix = cv2.getRotationMatrix2D((width_1 / 2, height_1 / 2), 270, 1)
            frame_1 = cv2.warpAffine(frame_1, matrix, (width_1, height_1))

        height_1, width_1, channels_1 = frame_1.shape

        # Detecting objects
        blob_1 = cv2.dnn.blobFromImage(frame_1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net_1.setInput(blob_1)
        outs_1 = net_1.forward(output_layers_1)

        class_ids_1 = []
        confidences_1 = []
        boxes_1 = []

        for out in outs_1:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width_1)
                    center_y = int(detection[1] * height_1)
                    w = int(detection[2] * width_1)
                    h = int(detection[3] * height_1)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes_1.append([x, y, w, h])
                    confidences_1.append(float(confidence))
                    class_ids_1.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes_1, confidences_1, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes_1)):
            if i in indexes:
                x, y, w, h = boxes_1[i]

                # api 결과 읽어오기
                f = open("../atoy/result.txt", 'r')
                result = f.readlines()
                f.close()
                f = open("../atoy/name.txt", 'r')
                names = f.readlines()
                f.close()
                f = open("../atoy/confidence.txt", 'r')
                similarity = f.readlines()
                f.close()

                # 좌표저장
                xy = [x, y]
                xytmp = 0
                for location in xylist_1:
                    if (location[0] < xy[0] + pixelgap) and (location[0] > xy[0] - pixelgap):
                        if (location[1] < xy[1] + pixelgap) and (location[1] > xy[1] - pixelgap):
                            location[0] = xy[0]
                            location[1] = xy[1]
                            xytmp = 1
                            if str(location[2]) + '\n' in result:
                                index = result.index(str(location[2]) + '\n')
                                information = names[index].rstrip('\n') + " " + similarity[index].rstrip('\n')
                                color = colors[i]
                                cv2.rectangle(frame_1, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(frame_1, information, (x, y + h + 16), font, 1.5, color, 2)

                if xytmp == 1:
                    continue
                else:
                    for tmp in xylist_1:
                        if tmp[2] == count_1:
                            del xylist_1[xylist_1.index(tmp)]
                    xy.append(count_1)
                    xylist_1.append(xy)

                    # 범위 안의 상의 부분 지정
                    if (y + 2 * h - 1) < frame_1.shape[0]:
                        top = (y + 2 * h - 1)
                    else:
                        top = frame_1.shape[0] - 1
                    if (y + 4 * h) < frame_1.shape[0]:
                        bottom = (y + 4 * h)
                    else:
                        bottom = frame_1.shape[0]
                    if (x - int(w / 3)) > 0:
                        left = x - int(w / 3)
                    else:
                        left = 0
                    if (x + w + int(w / 3)) < frame_1.shape[1]:
                        right = x + w + int(w / 3)
                    else:
                        right = frame_1.shape[1]

                    cloth = frame_1[top:bottom, left:right]
                    name = str(count_1) + ".jpg"
                    cv2.imwrite("../cloth/" + name, cloth)
                    if not cloth.any():
                        continue

                    # 일치하는 2개의 색이 없으면 continue
                    if len(whatcolor1) != 0:
                        colorflag = 0
                        top1, top2 = color_detection(cloth)
                        print(top1, top2)
                        if top1 in whatcolor1 or top2 in whatcolor1:
                            colorflag = 1

                        if colorflag == 0:
                            continue


                name = str(count_1) + ".jpg"
                faceimg = frame_1[y - 30:y + h + 30, x - 30:x + w + 30]
                if not faceimg.any():
                    continue
                faceimg = cv2.resize(faceimg, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
                # saving mask
                cv2.imwrite('../picture/' + name, faceimg)
                cv2.imwrite('/home/hogi/hdd/faceimg/' + name, faceimg)
                count_1 += 1

                # check color and save


        frame_1 = cv2.resize(frame_1, None, fx=2.0, fy=2.0)
        cv2.imshow("frame", frame_1)
        if cv2.waitKey(33) > 0:
            break

    cap_1.release()
    print("program end")


if __name__ == "__main__":
    start = time.time()
    color = []
    mainfunction1(0)
    print(time.time() - start)
