import cv2
import sqlite3


def getProfile(id):
    conn=sqlite3.connect("personinfo.db")
    cmd="select * from person where id="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer/trainingData.yml");
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img,"name: "+str(profile[1]),(x,y+h+50),font,4,(0,0,255),2,cv2.LINE_AA);
            cv2.putText(img,"age: "+str(profile[2]),(x,y+h+100),font,4,(0,0,255),2,cv2.LINE_AA);
            cv2.putText(img,"gender: "+str(profile[3]),(x,y+h+150),font,4,(0,0,255),2,cv2.LINE_AA);
            cv2.putText(img,"criminal: "+str(profile[4]),(x,y+h+200),font,4,(0,0,255),2,cv2.LINE_AA);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
