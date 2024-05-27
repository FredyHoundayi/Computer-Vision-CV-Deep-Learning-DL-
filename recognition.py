import tensorflow
import numpy as np
from tensorflow.keras.models import load_model
import cv2

import os
facedetect=cv2.CascadeClassifier("computer vision\Face Recognition System\Face Recognition System\haarcascade_frontalface_default.xml")

video=cv2.VideoCapture(0)
W=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
H=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video.set(3,640)
video.set(4,480)
fps=video.get(cv2.CAP_PROP_FPS) 

video_recognition=cv2.VideoWriter("C:\\Users\\fred\\DL\\computer vision\\Face Recognition System\\Face Recognition System\\recognition_video.mp4",cv2.VideoWriter.fourcc("M","P","G","4"),int(250 /fps),(W,H))

font=cv2.FONT_HERSHEY_COMPLEX
model=load_model("C:\\Users\\fred\\DL\\computer vision\\Face Recognition System\\model_multiclassesV1.h5")

def classe_images(label):
    if label==0:
        return "Jude"
    elif label==1:
        return "Elon"
    elif label==2:
        return "Natacha"
    elif label==3:
        return "Kevin"
    elif label==4:
        return "Sadio"
    elif label==5:
        return "Fredy"
    
    
while video.isOpened():
    succes,frame=video.read()
    faces=facedetect.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5,minSize=(30,30))
    if succes:
        for x,y,w,h in faces:
            new_face=frame[y:y+h,x:x+w]
            img=cv2.resize(new_face,(50,50))
            img=img[:,:,1]
            img=img.reshape(1,50,50,1)
            prediction=np.argmax(model.predict(img))
            print("prediction",prediction)
            print("Nom",classe_images(prediction))
            print(model.predict(img))
            prob=np.array(float(np.max(model.predict(img))))
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(0,255,0),-3)
            cv2.putText(frame,str(classe_images(prediction)),(x,y-10),font,0.75,(1,1,1),1,cv2.LINE_AA)
            """elif prediction>0.5:
               cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
               cv2.rectangle(frame,(x,y-40),(x+w,y),(0,255,0),-3)
               cv2.putText(frame,str(person_name(prediction)),(x,y-10),font,0.75,(255,255,255),1,cv2.LINE_AA)"""
            cv2.putText(frame,str(prob*100)+"%",(100,75),font,0.75,(255,100,100),2,cv2.LINE_AA)
           
        cv2.imshow("Recognition",frame)
        video_recognition.write(frame)
            
        if cv2.waitKey(int(1000/fps)) & 0xFF==ord("q"):
            break
        else:
            continue  

        
video.release()
video_recognition.release()
cv2.destroyAllWindows()
