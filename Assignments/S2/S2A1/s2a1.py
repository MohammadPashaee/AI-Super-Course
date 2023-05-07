# Mohammad Pashaee - Assignment 1 Season 2

import cv2
import numpy as np
import time

t0 = time.time()

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	
	if ret:
		f1 = cv2.flip(frame, 1)
		
		f2 = 255 - f1
		
		L = np.concatenate([f1, f2], 0)
		
		f3 = f1.copy()
		f3[:, :, 2] = 255
		
		g = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
		g1 = (g.reshape(480,640,1))
		gray = np.concatenate([g1, g1, g1], 2)
		
		R = np.concatenate([f3, gray], 0)
		
		H = np.concatenate([L, R], 1)
		
		t1 = time.time() - t0
		t1_str = str(round(t1, 2))
		
		cv2.putText(H, "Mohammad Pashaee", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 0, 255), 2)

		cv2.putText(H, t1_str, (50,100), cv2.FONT_HERSHEY_SIMPLEX,
			1, (0, 0, 255), 2)	
			
		cv2.imshow("Webcam", H)
		
		q = cv2.waitKey(1)     # age  bashe ta abad haminja gir mikone
		
		if q == ord('q'):
			break
			
cv2.destroyAllWindows()
cap.realease()

