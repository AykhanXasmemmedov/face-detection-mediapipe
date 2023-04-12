import cv2
import time
import mediapipe as mp

# mediapipe face detection is creating and determining
mp_face_detection=mp.solutions.face_detection
Face_detection=mp_face_detection.FaceDetection() # DEFAULT VALUE IS O.5,FOR ACCURATE VALUE WRITE 0.75
# drawing_utils face lines are drawing
mpDraw=mp.solutions.drawing_utils

camera=cv2.VideoCapture(0)

ptime=0
while True:
    success,video=camera.read()
    video=cv2.flip(video,1)

    videoRGB=cv2.cvtColor(video,cv2.COLOR_BGR2RGB)
    result=Face_detection.process(videoRGB)

   
    h,w,c=video.shape
    #face_number=0
    if result.detections:
        #print(result.detections)
        for id, detection in enumerate(result.detections):
            #face_number=face_number+1
            
            #print(id,detection)

            coordinate=mpDraw.draw_detection(video,detection)
            print(coordinate)
            bboxCl=detection.location_data.relative_bounding_box
            bbox=int(bboxCl.xmin*w),int(bboxCl.ymin*h),int(bboxCl.width*w),int(bboxCl.height*h)
            
            #cv2.rectangle(video,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)
            #cv2.rectangle(video,bbox,(0,255,0),2)
            
            
            
            cv2.putText(video,f"confidence:{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-10),cv2.FONT_ITALIC,1,(0,255,0),2)
        print(f"faces number:{id+1}")


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