import cv2
import numpy as np
from ultralytics import YOLO


num_car = 0
num_bike = 0
y_lim1 = 260
y_lim2 = 320
Center_frame0 = [[0, 0]]

model = YOLO("yolov8x")
# model = YOLO("yolov8n")

cap = cv2.VideoCapture('Road.mp4')

while True:
    ret, frame = cap.read()

    if ret:
        
        # frame_New = frame[350:550, 200:900]
        # height, width = frame.shape[:2]

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        cv2.line(frame, (0, y_lim1), (850,y_lim1), (0,0,255), 2)
        cv2.line(frame, (0, y_lim2), (850,y_lim2), (0,0,255), 2)

        frame_New = frame[y_lim1:y_lim2, 0:850]

        results = model.predict(source=frame_New, show=False)

        centers = []
        
        for result in results:
            for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):
                obj_cls = int(obj_cls)

                if obj_cls == 2:      # Car ID
                    xc1 = obj_xyxy[0].item()
                    yc1 = obj_xyxy[1].item()
                    xc2 = obj_xyxy[2].item()
                    yc2 = obj_xyxy[3].item()

                    center = [(xc1+xc2)/2, (yc1+yc2)/2]
                    centers.append(center)
                    
                    p1 = np.array(center)
                    d= []

                    cv2.rectangle(frame_New, (int(xc1), int(yc1)), (int(xc2), int(yc2)), (255, 0, 0), 1)

                    if len(Center_frame0) == 0:
                         Center_frame0 = [[0, 0]]

                    for p in Center_frame0:
                         p2 = np.array(p)
                         d1 = np.linalg.norm(p1 - p2)
                         d.append(d1)

                    if min(d) < 20:
                         same_object = True
                    else:
                         same_object = False

                    if not(same_object):
                         num_car += 1
                    

                elif obj_cls == 3:    # Bike ID
                    xb1 = obj_xyxy[0].item()
                    yb1 = obj_xyxy[1].item()
                    xb2 = obj_xyxy[2].item()
                    yb2 = obj_xyxy[3].item()

                    center = [(xb1+xb2)/2, (yb1+yb2)/2]
                    centers.append(center)

                    p1 = np.array(center)
                    d= []

                    cv2.rectangle(frame_New, (int(xb1), int(yb1)), (int(xb2), int(yb2)), (255, 0, 0), 1)

                    if len(Center_frame0) == 0:
                        Center_frame0 = [[0, 0]]

                    for p in Center_frame0:
                        p2 = np.array(p)
                        d1 = np.linalg.norm(p1 - p2)
                        d.append(d1)

                    if min(d) < 20:
                        same_object = True
                    else:
                        same_object = False
                    
                    if not(same_object):
                        num_bike += 1
        
        Center_frame0 = centers.copy()

        cv2.putText(frame, f"Cars: {num_car} -- Motorcycles: {num_bike}", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow("Traffic Counter", frame)

        cv2.waitKey(1)

print(f"Total number of cars and biker are {num_car}, and {num_bike} repectively")

cv2.destroyAllWindows()