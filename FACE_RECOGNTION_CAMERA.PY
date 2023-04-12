import cv2
import time
import mediapipe as mp
import numpy as np
import os
NAME_PATH=r"C:\Users\ayxan\Pictures\face_projction\face_recogntion\face2"
names=[]
for name in os.listdir(NAME_PATH):
    names.append(name)

mp_face_detection=mp.solutions.face_detection
Face_detection=mp_face_detection.FaceDetection() # DEFAULT VALUE IS O.5,FOR ACCURATE VALUE WRITE 0.75

mpDraw=mp.solutions.drawing_utils
camera=cv2.VideoCapture(1)

features=np.load("median_features.npy",allow_pickle=True)
labels=np.load("median_labels.npy")

facerecogntion=cv2.face.LBPHFaceRecognizer_create()
facerecogntion.read("median_train.yml")

ptime=0
while True:
    success,video=camera.read()
    #video=cv2.flip(video,1)
    videoRGB=cv2.cvtColor(video,cv2.COLOR_BGR2RGB)
    result=Face_detection.process(videoRGB)
    #print(result)
    ih,iw,ic=video.shape
    face_number=0
    if result.detections:
        #print(result.detections)
        for id, detection in enumerate(result.detections):
            face_number=face_number+1
            
            #print(id,detection)
            #mpDraw.draw_detection(video,detection)
            bboxCl=detection.location_data.relative_bounding_box
            
            bbox=int(bboxCl.xmin*iw),int(bboxCl.ymin*ih),int(bboxCl.width*iw),int(bboxCl.height*ih)
            x,y,w,h=bbox
            face=video[y:y+h,x:x+w]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            label,confidence=facerecogntion.predict(face)

            #cv2.rectangle(video,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)
            cv2.rectangle(video,bbox,(0,255,0),2)
            
            print(confidence)
            if confidence>10:
                cv2.putText(video,f"name:{names[label]}",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),1)
            else:
                cv2.putText(video,"unknown",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),1)

                        
            #cv2.putText(video,f"confidence:{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-10),cv2.FONT_ITALIC,1,(0,255,0),2)


    #print(detection.score)
    #print(f"faces number:{face_number}")
    ctime=time.time()
    FPS=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(video,f"FPS: {int(FPS)}",(40,40),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    cv2.imshow("video",video)

    key=cv2.waitKey(27)
    if key==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()