import cv2

cap = cv2.VideoCapture("../test.mp4")

name = 0
while True:
    ret, frame = cap.read()

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        print("video end")
        break

    cv2.imwrite("../picture/" + str(name) + ".jpg", frame)
    name += 1
