import cv2
import numpy as np

# Loading Image

img = cv2.imread("Balls.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel_sharp = np.array([[0, -1, 0],
# 	[-1, 5, 0],
# 	[0, -1, 0]])
# sharp = cv2.filter2D(img, -1, kernel_sharp)

# blur = cv2.GaussianBlur(gray, (3, 3), 0)
# edge = cv2.Canny(sharp, 140, 150)
# circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1.56, 330, maxRadius=1200)

circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5 ,minDist=200 ,param1=100, param2=50, maxRadius=100, minRadius=70,)

num_ball = circles.shape[1]
print(f"Total number of balls = {num_ball}")

if  circles is not None:
    circles = circles[0].astype(np.uint32)

    for circle in circles:
        cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
        
        if 410 < circle[0] < 600 and 855 < circle[1] < 1050:
            x_red = circle[0]
            y_red = circle[1] 
            
            print(f"The position of the red ball is = ({x_red}, {y_red})")

# Displaying Image

img = cv2.resize(img, None, fx=0.5, fy=0.5) 

cv2.imshow("Image", img)

cv2.waitKey(0)

cv2.destroyAllWindows()