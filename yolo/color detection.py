import cv2
import numpy as np
import time

show_picture = 1
size = 200
picture = '2'

white = 0
black = 0
red = 0       # (166 ~ 180, 0 ~ 15)
yellow = 0     # (16 ~ 45)
green = 0      # (46 ~ 75)
blue = 0     # (76 ~ 135)
purple = 0     # (136 ~ 165)
background = 0

start = time.time()
src = cv2.imread("../cloth/" + picture + ".jpg", cv2.IMREAD_COLOR)
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

print(black, white, red, yellow, green, blue, purple, background)

color = [black, white, red, yellow, green, blue, purple]
colortext = {0: 'black', 1: 'white', 2: 'red', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'purple'}
# 리스트 참조가 아닌 복사
tmp = color[:]
tmp.sort(reverse=True)

print(colortext[color.index(tmp[0])], colortext[color.index(tmp[1])])

print(time.time()-start)

if show_picture is 1:
    rl = np.array([0, 1, 1])
    rh = np.array([180, 70, 70])
    blackmask = cv2.inRange(hsv, rl, rh)

    rl = np.array([0, 1, 190])
    rh = np.array([180, 70, 255])
    whitemask = cv2.inRange(hsv, rl, rh)

    rl = np.array([0, 70, 0])
    rh = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, rl, rh)
    rl = np.array([166, 70, 0])
    rh = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, rl, rh)
    redmask = cv2.bitwise_or(mask1, mask2)

    rl = np.array([11, 70, 0])
    rh = np.array([35, 255, 255])
    yellowmask = cv2.inRange(hsv, rl, rh)

    rl = np.array([36, 70, 0])
    rh = np.array([75, 255, 255])
    greenmask = cv2.inRange(hsv, rl, rh)

    rl = np.array([76, 70, 0])
    rh = np.array([125, 255, 255])
    bluemask = cv2.inRange(hsv, rl, rh)

    rl = np.array([126, 70, 0])
    rh = np.array([165, 255, 255])
    purplemask = cv2.inRange(hsv, rl, rh)

    rl = np.array([0, 0, 0])
    rh = np.array([180, 0, 0])
    backgroundmask = cv2.inRange(hsv, rl, rh)

    pic1 = cv2.bitwise_and(hsv, hsv, mask=blackmask)
    pic1 = cv2.cvtColor(pic1, cv2.COLOR_HSV2BGR)
    pic2 = cv2.bitwise_and(hsv, hsv, mask=whitemask)
    pic2 = cv2.cvtColor(pic2, cv2.COLOR_HSV2BGR)
    pic3 = cv2.bitwise_and(hsv, hsv, mask=redmask)
    pic3 = cv2.cvtColor(pic3, cv2.COLOR_HSV2BGR)
    pic4 = cv2.bitwise_and(hsv, hsv, mask=yellowmask)
    pic4 = cv2.cvtColor(pic4, cv2.COLOR_HSV2BGR)
    pic5 = cv2.bitwise_and(hsv, hsv, mask=greenmask)
    pic5 = cv2.cvtColor(pic5, cv2.COLOR_HSV2BGR)
    pic6 = cv2.bitwise_and(hsv, hsv, mask=bluemask)
    pic6 = cv2.cvtColor(pic6, cv2.COLOR_HSV2BGR)
    pic7 = cv2.bitwise_and(hsv, hsv, mask=purplemask)
    pic7 = cv2.cvtColor(pic7, cv2.COLOR_HSV2BGR)

    cv2.imshow("original", src)
    cv2.imshow("black", pic1)
    cv2.imshow("white", pic2)
    cv2.imshow("red", pic3)
    cv2.imshow("yellow", pic4)
    cv2.imshow("green", pic5)
    cv2.imshow("blue", pic6)
    cv2.imshow("purple", pic7)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
