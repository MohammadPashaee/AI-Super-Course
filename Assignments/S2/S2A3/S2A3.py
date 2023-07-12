import vlc
import cv2
import numpy as np
import time

A = input("Choices:\n1. Movie\n2. Music\n3. Book\nYour Decision (Between 1 to 3):")

if A != '1' and A != '2' and A != '3':
    print("Choose a number between 1 to 3")
    quit()

B = input("Face Property:\n1. Normal\n2. Blured (Between 1 or 2):")

if B != '1' and B != '2':
    print("Choose a number between 1 or 2")
    quit()

class MyMP():
    def __init__(self):
        self.fdetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.edetector = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.cap = cv2.VideoCapture(0)

    def show_poster(self):
        img = cv2.imread("Poster.jpg")
        img = cv2.resize(img, (640,720))
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def play_music(self, target_dir):
        music = vlc.MediaPlayer(target_dir)
        music.play()
        time.sleep(2)

    def write_book(self):
        print("Recommended Book = Kafka on the Shore")

    def stream_webcam(self):
        while True:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.flip(frame, 1)

                fresults = self.fdetector.detectMultiScale(frame)

                for (x, y, w, h) in fresults:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                    
                    face = frame[y:y+h, x:x+w]
                    
                    if B == '1':
                        eresults = self.edetector.detectMultiScale(face)

                        for (ex, ey, ew, eh) in eresults:
                            cv2.rectangle(face, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                    
                    elif B == '2':
                        
                        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face, (51, 51), 0)
                        

                cv2.imshow("Webcam", frame)

                q = cv2.waitKey(1)

                if q == ord('q'):
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()

mp = MyMP()


if A == '1':
    mp.show_poster()
    mp.stream_webcam()
elif A == '2':
    mp.play_music("Drop.mp3")
    mp.stream_webcam()
elif A == '3':
    mp.write_book()
    mp.stream_webcam()