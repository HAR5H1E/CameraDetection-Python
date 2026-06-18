import cv2 as cv 
import numpy as np
import mediapipe as mp
import time
import osascript
import math
import threading
import osascript


latestLandMark = None
gestureLandmark = None
prevVol = None
VolThread = None

cap = cv.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def volumeControl(volume):
    global prevVol

    if prevVol is None:
        osascript.osascript(f"set volume output volume {round(volume)}")
        prevVol = round(volume) 
    elif prevVol != round(volume):
        osascript.osascript(f"set volume output volume {round(volume)}")
        prevVol = round(volume)
    time.sleep(0.05)
    


def GetResult(results,output,timestamp):
    global latestLandMark,gestureLandmark
    if results.gestures and results.hand_landmarks:
        latestLandMark = results.hand_landmarks
        gestureLandmark = results.gestures[0][0].category_name
    else:
        latestLandMark = None
        gestureLandmark = None

options = GestureRecognizerOptions(
            base_options = BaseOptions("./Task/gesture_recognizer.task"),
            running_mode = VisionRunningMode.LIVE_STREAM,
            num_hands = 1,
            result_callback = GetResult
)

recog = GestureRecognizer.create_from_options(options)

#Camera Warmup
for _ in range(30):
    cap.read()

stopMusic = False
while True:
    succes , Frame = cap.read()

    if cap.isOpened() == False:
        print("error: No Camera")
        break

    if not succes:
        print("Can't get frame")
        break

    frameShow = Frame
    frameShow = cv.flip(frameShow,1)
    Frame = cv.flip(Frame,1)
    Frame = cv.cvtColor(Frame,cv.COLOR_BGR2RGB)

    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data = Frame)
    
    currTime = int(time.time_ns()/1000000)
    recog.recognize_async(mpImage,currTime)


    if latestLandMark and gestureLandmark:
        pixel = []
        h,w,_ = frameShow.shape
        for lat in latestLandMark:

          

            Thumb = lat[4]
            Finger = lat[8]

            tx,ty = int(Thumb.x * w),int(Thumb.y * h)
            fx,fy = int(Finger.x * w), int(Finger.y * h)

            cv.circle(frameShow,(tx,ty),4,(0,255,0),cv.FILLED)
            cv.circle(frameShow,(fx,fy),4,(0,255,0),cv.FILLED)
            cv.line(frameShow,(fx,fy),(tx,ty),(0,255,0),3)

            diff = math.sqrt((fx - tx)**2 + (fy - ty)**2)


            volume = np.interp(diff,[30,160],[0,100])

            
            if stopMusic:

                volume = 0
            
            
            if  (VolThread is None or not VolThread.is_alive()):

                VolThread = threading.Thread(target=volumeControl,args =(volume,))
                VolThread.start()


            for lm in lat:
                lx = int(lm.x * w)
                ly = int(lm.y * h)

                pixel.append([lx,ly])
        
            pixel_arr = np.array(pixel,dtype=np.int32)


            (xx,yy,bw,bh)=cv.boundingRect(pixel_arr)

            #cv.rectangle(frameShow,(xx,yy),(xx+bw,yy+bh),(0,255,0),3)
            cv.putText(frameShow,gestureLandmark,(xx,yy - 10),cv.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
            if gestureLandmark == "Closed_Fist":
                
                stopMusic = True
            else:

                stopMusic = False

        


    cv.imshow("Detector", frameShow)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


