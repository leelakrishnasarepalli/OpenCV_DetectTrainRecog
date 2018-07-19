
import os
import cv2
import numpy as np
from PIL import Image

rec = cv2.face.LBPHFaceRecognizer_create();

path = 'dataSet'

def getImagesWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePath in imagePaths:
        if(imagePath!='dataSet/.DS_Store'):
            faceImg=Image.open(imagePath).convert('L');
            faceNP=np.array(faceImg,'uint8')
            id=int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNP)
            ids.append(id)
            cv2.imshow("training",faceNP)
            cv2.waitKey(10)
    return np.array(ids),faces
    
    
ids,faces = getImagesWithID(path)
rec.train(faces,ids)
rec.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()
