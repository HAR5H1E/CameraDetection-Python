import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

first_frame = None
if not cap.isOpened():
    print("Can't Open Camera")
    exit()

for i in range(100):
        cap.read()

print("RUN")
while True:
    

    
    ret , frame = cap.read()


    if not ret:
        print("Can't recieve")
        break

    

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

    #cv.accumulateWeighted(gray,first_frame,0.03)
    #background = cv.convertScaleAbs(first_frame)

    absDiff = cv.absdiff(first_frame,gray)

    threshHold = cv.threshold(absDiff,60,255,cv.THRESH_BINARY)[1]
    threshHold = cv.dilate(threshHold,None,iterations=3)

    (cnts,_)= cv.findContours(threshHold.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cn in cnts:
        if cv.contourArea(cn)<3000:
            continue
        (x,y,w,h) = cv.boundingRect(cn)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    #cv.imshow("Delta",absDiff)
    cv.imshow("ThresHold",threshHold)
    #cv.imshow("Frame",gray )


    
    cv.imshow("Feed",frame)

    if cv.waitKey(1) == ord('q'):
        break

    print(threshHold)

cap.release()
cv.destroyAllWindows()
