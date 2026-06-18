import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import osascript

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


latest = None

def print_result(result, output_image, timestamp_ms):
    global latest
    if result.hand_landmarks:
        latest = result.hand_landmarks
    else:
        latest = None


cap = cv.VideoCapture(0)
for i in range(30):
    cap.read()

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "./Task/hand_landmarker.task"),
    running_mode = vision.RunningMode.LIVE_STREAM,
    num_hands = 1,
    min_hand_detection_confidence = 0.5,
    
    result_callback = print_result
    
)

detector = vision.HandLandmarker.create_from_options(options)



while True:
    if cap.isOpened() == False:
        print("Failed capture")
        break

    scores , frame = cap.read()
    frame = cv.flip(frame,1)
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    currTime = int(time.time_ns()/1000000)

    detector.detect_async(mp_Image,currTime)

    prevVol = -1

    if latest is not None:
        h,w,_ = frame.shape
        for lat in latest:

            thumbPoint = lat[4]
            FingPoint = lat[8]

            tx,ty = int(thumbPoint.x*w),int(thumbPoint.y*h)
            fx,fy = int(FingPoint.x*w),int(FingPoint.y*h)

            cv.circle(frame,(tx,ty),5,(0,255,0),cv.FILLED)
            cv.circle(frame,(fx,fy),5,(0,255,0),cv.FILLED)
            cv.line(frame,(fx,fy),(tx,ty),(0,255,0),2)

            pixDis = math.sqrt((fx-tx)**2 + (fy-ty)**2)
            Vol = np.interp(pixDis,[20,160],[0,100])
            
            if Vol != prevVol:
                osascript.osascript(f"set volume output volume {Vol}")
                prevVol = Vol


   
    cv.imshow("Starter",frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()