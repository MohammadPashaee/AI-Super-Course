import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8x")

kernel = np.array([[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]])  

for i in range(1,7):
    img = cv2.imread(f"{i}.jpg")

    img = cv2.resize(img, (1280, 720))

    img = img[300:720, 450:1000]

    img_show = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(gray, 100, 120)

    lines = cv2.HoughLines(edge, 1, np.pi/180, 200)

    x1=[]
    y1=[]
    x2=[]
    y2=[]
    
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr

        a=np.cos(theta)
        b = np.sin(theta)

        x0 = a*r
        y0 = b*r

        xx1 = int(x0+1000*(-b))
        yy1 = int(y0+1000*(a))
        xx2 = int(x0-1000*(-b))
        yy2 = int(y0-1000*(a))

        x1.append(xx1)
        y1.append(yy1)
        x2.append(xx2)
        y2.append(yy2)    

    x1_line = np.max(x1)
    y1_line = np.min(y1)
    x2_line = np.min(x2)
    y2_line = np.max(y2)
    
    cv2.line(img_show, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 2)

    m = (y2_line - y1_line) / (x2_line - x1_line)

    b = -m * x1_line + y1_line
    
    results = model.predict(source=img)

    for result in results:
        for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):
            obj_cls = int(obj_cls)

            if obj_cls == 32:
                xb1 = obj_xyxy[0].item()
                yb1 = obj_xyxy[1].item()
                xb2 = obj_xyxy[2].item()
                yb2 = obj_xyxy[3].item()

                ball_x = (xb2+xb1)/2
                ball_y = (yb2+yb1)/2

                ball_r = ((xb2-xb1)+(yb2-yb1))/4

                yhat = ball_x * m + b

                if yhat > ball_y:
                    goal = False
                else:
                    distance = abs(m*ball_x-ball_y+b)/(m**2+1)**0.5
                    print(distance, ball_r)
                    if distance > ball_r:
                        goal = True
                        break

                cv2.rectangle(img_show, (int(xb1), int(yb1)), (int(xb2), int(yb2)), (0, 0, 255), 2)
        if goal:
            break


    cv2.imshow("Frames", img_show)

    cv2.waitKey(0)

    if goal:
        break

cv2.destroyAllWindows()

if goal:
    print("Goaaaaaaaaaaaaaaaal!!!")