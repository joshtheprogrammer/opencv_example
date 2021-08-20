import cv2 as cv
import numpy as np
from PIL import Image
import os

path = 'C:/Users/joshu/Desktop/projects/detection/2/dataset/' 
recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier("C:/Users/joshu/Desktop/projects/detection/2/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print(imagePaths)
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        try: 
            for image in os.listdir(imagePath):
                PIL_img = Image.open(os.path.join(imagePath, image)).convert('L')
                img_numpy = np.array(PIL_img,'uint8')
                id = int(os.path.split(image)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
        except:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('C:/Users/joshu/Desktop/projects/detection/2/trainer/trainer.yml') 

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))