import urllib.request as req

import cv2 as cv
import numpy as np

#import os 

def urlIMG(url):
    res = req.urlopen(url)
    img_arr = np.asarray(bytearray(res.read()), dtype="uint8")

    return cv.imdecode(img_arr, -1)

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/joshu/Desktop/projects/detection/2/trainer/trainer.yml')
faceCascade = cv.CascadeClassifier("C:/Users/joshu/Desktop/projects/detection/2/haarcascade_frontalface_default.xml")
font = cv.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'joshua', 'my dad'] 
people = ['None']

joshua = False
joshua2 = False

noob = cv.imread("C:/Users/joshu/Desktop/projects/detection/2/placeholder.png")
sus = urlIMG("https://imgix.bustle.com/uploads/image/2020/10/31/31aa14f0-bc99-4e6b-b785-26ae420971dd-screen-shot-2020-10-31-at-52151-pm.png?w=1200&h=630&fit=crop&crop=faces&fm=jpg")

cam = cv.VideoCapture(0)

cam.set(3, 1024) # width
cam.set(4, 1024) # height

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

fourcc = cv.VideoWriter_fourcc(*'XVID')  
out = cv.VideoWriter('Videos/output.avi',fourcc, 10.0, (int(cam.get(3)),int(cam.get(4))))  

while True:
    ret, frame = cam.read()
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minW), int(minH)),)
        for (x,y,w,h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence <= 80):
                if (confidence <= 65):
                    id = names[id]
                else:
                    id = str(names[id] + "?")
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "N/A"
                confidence = "  {0}%".format(round(100 - confidence))

            cv.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

            #detect if joshua is in the picture, maybe put people in array[]
            people.append(id)

            sus2scale = cv.resize(sus, (25, 25), interpolation=cv.INTER_AREA)
            s_w, s_h, _ = sus2scale.shape
            noob2scale = cv.resize(noob, (25, 25), interpolation=cv.INTER_AREA)
            n_w, n_h, _ = noob2scale.shape

            if joshua == True:
                if id == names[1]:
                    c = (255,0,0)
                    try:
                        frame[y:y+n_h, x-30:x-30+n_w] = noob2scale
                        frame[y+30:y+30+n_h, x-30:x-30+n_w] = sus2scale
                    except:
                        pass
                if id != names[1]:
                    c = (0,0,255)
                    try:
                        frame[y:y+n_h, x-30:x-30+n_w] = sus2scale
                    except:
                        pass
            elif joshua2 == True:
                if id == str(names[1]+"?"):
                    c = (0,150,150)
                    try:
                        frame[y:y+n_h, x-30:x-30+n_w] = noob2scale
                        frame[y+30:y+30+n_h, x-30:x-30+n_w] = sus2scale
                    except:
                        pass
                if id != str(names[1]+"?"):
                    c = (0,0,255)
                    try:
                        frame[y:y+n_h, x-30:x-30+n_w] = sus2scale
                    except:
                        pass
            else:
                c = (0,255,0)

            cv.rectangle(frame, (x,y), (x+w,y+h), c, 2)

        for p in people:
            if p == names[1]:
                joshua = True
            else:
                joshua = False
                
            if p == str(names[1] + "?"):
                joshua2 = True
            else:
                joshua2 = False

        people = []

        out.write(frame)

        cv.imshow('video',frame)
        
        k = cv.waitKey(10) & 0xff
        if k == 27:
            break
    else:
        break

print("\n [INFO] Exiting Program and cleanup stuff, saving video")
cam.release()
out.release()
cv.destroyAllWindows()