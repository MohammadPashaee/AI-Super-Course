import cv2
import numpy as np
import time

fdetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
edetector = cv2.CascadeClassifier("haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)

        fresults = fdetector.detectMultiScale(frame)
        
        for (x, y, w, h) in fresults:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            
            face = frame[y:y+h, x:x+w]
            eresults = edetector.detectMultiScale(face)
            
            for (ex, ey, ew, eh) in eresults:
                cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        cv2.imshow("Webcam", frame)

        q = cv2.waitKey(1)

        if q == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()     