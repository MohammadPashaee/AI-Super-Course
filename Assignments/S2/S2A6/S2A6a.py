import cv2
import numpy as np

lx1 = 220.
ly1 = 170.
lx2 = 400.
ly2 = 330.

m = (ly2 - ly1) / (lx2 - lx1)

b = -m * lx1 + ly1

kernel = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]])    

for i in range(1,7):
    img = cv2.imread(f"{i}.jpg")

    img = cv2.resize(img, (1280, 720))

    img = img[300:720, 450:1000]

    img_show = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharp = cv2.filter2D(gray, -1, kernel)

    edge = cv2.Canny(sharp, 100, 120)

    circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1 ,minDist=100 ,param1=2, param2=12, minRadius=7, maxRadius=11)
    
    if  circles is not None:
        circles = circles[0].astype(np.uint32)

        for circle in circles:
            cv2.circle(img_show, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
             
            yhat = circle[0] * m + b

            if yhat > circle[1]:
                goal = False
            else:
                distance = abs(m*circle[0]-circle[1]+b)/(m**2+1)**0.5
                print(distance, circle[2])
                if distance > circle[2]:
                    goal = True
                    break
        if goal:
            break

    cv2.imshow("Image", img_show)

    cv2.waitKey(0)

    if goal:
        break

cv2.destroyAllWindows()

if goal:
    print("GOAAAAAAAAAAAAL!!")