import numpy as np
import os
import cv2
import mediapipe as mp

face_detection=mp.solutions.face_detection.FaceDetection()

path=r"C:\Users\ayxan\Pictures\face_projction\test_imageS"
images_folder=[]
for images in os.listdir(path):
    images_folder.append(images)
print(images_folder)

NAME_PATH=r"C:\Users\ayxan\Pictures\face_projction\face_recogntion\face1"
names=[]
for name in os.listdir(NAME_PATH):
    names.append(name)


features=np.load("median_features.npy",allow_pickle=True)
labels=np.load("median_labels.npy")

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("median_train.yml")

for image_name in images_folder:
    image_path=os.path.join(path,image_name)
    image=cv2.imread(image_path)
    imageBGR=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result=face_detection.process(imageBGR)
    ih,iw=image.shape[:2]
    
    if result.detections:
        for id,detection in enumerate(result.detections):
            #print(image_name)
            print(id,detection)
            bboxcl=detection.location_data.relative_bounding_box
            bbox=int(bboxcl.xmin*iw),int(bboxcl.ymin*ih),int(bboxcl.width*iw),int(bboxcl.height*ih)
            x,y,w,h=bbox
            x1,y1=x+w,y+h
            face=image[y:y1,x:x1]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            label,confidence=face_recognizer.predict(face)
            cv2.rectangle(image,(x,y),(x1,y1),(0,255,0),2)
            #cv2.putText(image,f"conf:{confidence}",(x-10,y-10),cv2.FONT_ITALIC,1.0,(0,255,0),1)
            cv2.putText(image,f"{names[label]}",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),1)
    cv2.imshow("image",image)
    cv2.imshow("face",face)

    cv2.waitKey(0)
    
