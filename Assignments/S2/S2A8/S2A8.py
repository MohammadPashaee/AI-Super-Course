import cv2
import numpy as np
import mediapipe as mp

f= (np.array([[255, 255, 255],[255, 255, 255]])).reshape((2,3,1)).astype(np.uint8)
frame = np.concatenate([f, f, f], 2)
frame = cv2.resize(frame, (566,715))

x_eye_desired = 68.5601476430893      # Number of pixels between eyes in x-dimension

model = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5)


for i in range(1,13):
    img = cv2.imread(f"imgs/{i}.jpg")

    width, height = img.shape[1], img.shape[0]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lms = model.process(img_rgb).multi_face_landmarks

    if lms:
        for lm in lms:
            eye_left = [lm.landmark[33].x, lm.landmark[33].y]
            eye_right = [lm.landmark[359].x, lm.landmark[359].y]

            x_eye_length = eye_right[0]*width - eye_left[0]*width

            # cv2.circle(img, (int(eye_left[0]*width), int(eye_left[1]*height)), 3, (0, 0, 255), cv2.FILLED)
            # cv2.circle(img, (int(eye_right[0]*width), int(eye_right[1]*height)), 3, (0, 0, 255), cv2.FILLED)

    ratio_x = x_eye_desired/ x_eye_length

    width_new = int(width * ratio_x)
    height_new = int(height * ratio_x)
    x_eye_right_new = int((eye_right[0] * width) * ratio_x)
    y_eye_right_new = int((eye_right[1] * height) * ratio_x)

    img_resized = cv2.resize(img, (width_new, height_new), interpolation = cv2.INTER_AREA)
    img_new = img_resized[y_eye_right_new - 100 : y_eye_right_new + 190, x_eye_right_new - int(x_eye_desired) - 98 : x_eye_right_new+94, :]

    frame [210:500, 153:413] = img_new

    cv2.imshow("Image", frame)

    q = cv2.waitKey(200)

    if q == ord('q'):
        break

cv2.destroyWindow("Image")