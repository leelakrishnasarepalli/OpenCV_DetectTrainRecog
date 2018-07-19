import cv2
import sqlite3

def insertOrUpdate(id,name):
    conn=sqlite3.connect("personinfo.db")
    cmd="select * from person where id="+str(id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="update person set name="+str(name)+" where id="+str(id)
    else:
        cmd="insert into person(id,name) values("+str(id)+","+str(name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);

id=input('enter user id')
name=input('enter user name')
insertOrUpdate(id,name)
sampleNum=0;
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    if(sampleNum>20):
        break
cam.release()
cv2.destroyAllWindows()
