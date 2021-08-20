import cv2 as cv
import numpy as np
import os

from numpy.lib.function_base import place

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_detector = cv.CascadeClassifier("C:/Users/joshu/Desktop/projects/detection/2/haarcascade_frontalface_default.xml")

face_id = input('\n enter user id end press <return> ==>  ')
#add file (eg.. 2, 3) if no value
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0
imgs = []

while(True):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        if not os.path.exists('C:/Users/joshu/Desktop/projects/detection/2/dataset/{}'.format(face_id)):
            os.makedirs('C:/Users/joshu/Desktop/projects/detection/2/dataset/{}'.format(face_id))
        cv.imwrite("C:/Users/joshu/Desktop/projects/detection/2/dataset/{}/User.".format(face_id) + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        print('C:/Users/joshu/Desktop/projects/detection/2/dataset/{}/User.{}.jpg'.format(str(face_id), str(face_id) + "." + str(count)))
        imgs.append('C:/Users/joshu/Desktop/projects/detection/2/dataset/{}/User.{}.jpg'.format(str(face_id), str(face_id) + "." + str(count)))

    cv.imshow("frame", frame)

    k = cv.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv.destroyAllWindows()

__ = []
for col in range(0, 6):
    _ = []
    for row in range(0+(col*5), 5+(col*5)):
        placeholder = cv.imread("C:/Users/joshu/Desktop/projects/detection/2/placeholder.png")
        try:
            img = cv.imread(imgs[row])
            img = cv.resize(img, (placeholder.shape[1], placeholder.shape[0]), interpolation=cv.INTER_AREA)
        except Exception as E:
            img = placeholder
            print(E)
        finally:
            _.append(img)
    __.append(_)

h = [np.zeros((100, 100, 3), np.uint8)]*len(__)
for x in range(0, len(__)):
    h[x] = np.hstack(__[x])
v = np.vstack(h)

while True:
    cv.imshow("User", v)
    k = cv.waitKey(100) & 0xff
    if k == 27:
        break




