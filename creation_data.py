import cv2
import os

video=cv2.VideoCapture(0)

fps=video.get(cv2.CAP_PROP_FPS)   
         
path="C:\\Users\\fred\\DL\\computer vision\\Face Recognition System\\Face Recognition System\\images"
facedetect=cv2.CascadeClassifier("C:\\Users\\fred\\DL\\computer vision\\Face Recognition System\\Face Recognition System\\haarcascade_frontalface_default.xml")
count=0
nom=input("Entrer nom: ")
new_path=path+"/"+nom
existance=os.path.exists(new_path)

if existance:
    print("Changer le nom:Le dossier existe dejà .")
    nom=input("Entrer nom")
else:
    os.makedirs(new_path)
while video.isOpened():
    succes,frame=video.read()
    faces=facedetect.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5,minSize=(30,30))
    if succes==True:
        for x,y,w,h in faces:
           name=path+"/"+nom+"/"+nom+str(count+1)+".jpg" 
           print("Creating ..........."+name)
           cv2.imwrite(name,frame[y:y+h,x:x+w])
           cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
           count+=1   
        cv2.imshow("Image Collection",frame)
        if count > 10 :
            break
        if cv2.waitKey(int(1000/fps)) & 0xFF==ord("q"):
            break
        else:
            continue 
        
video.release()
cv2.destroyAllWindows()
