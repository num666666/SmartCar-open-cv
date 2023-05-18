import cv2 as cv
cap = cv.VideoCapture(0)
NoDrive = cv.imread("noDrive.jpg")
pedestrain = cv.imread("pedestrain.jpg")
NoDrive = cv.resize(NoDrive, (64,64))
pedestrain = cv.resize(pedestrain, (64,64))
NoDrive = cv.inRange(NoDrive, (89,91,149), (255,255,255))
pedestrain = cv.inRange(pedestrain, (89,91,149), (255,255,255))
cv.imshow("Nodrive", NoDrive)
cv.imshow("pedestrain", pedestrain)
while (True):
    ret, frame = cap.read()
    frameCopy = frame.copy()
    
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv,(5,5))
    mask = cv.inRange(hsv, (89,123,73), (255,255,255))
    
    mask = cv.erode(mask,None, iterations=2)
    mask = cv.dilate(mask, None, iterations = 4)
    cv.imshow("Mask", mask)
    
    contours, g = cv.findContours(mask,cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
   
    if contours:
        contours = sorted(contours, key = cv.contourArea, reverse = True)
        cv.drawContours(frame,contours[0], 0, (255,0,255),3)
        cv.imshow("Contours",frame)
        (x,y,w,h) = cv.boundingRect(contours[0])
        cv.rectangle(frame, (x,y), (x+w, y + h), (0,255,0),2)
        cv.imshow("Rect", frame)
        roImg = frameCopy[y:y+h, x:x+w]
        cv.imshow("Detect", roImg)
        roImg = cv.resize(roImg, (64,64))
        roImg = cv.inRange(roImg, (89,91,149), (255,255,255))
        cv.imshow("detect1", roImg)
        noDrive_val = 0
        pedestrain_val = 0
        for i in range(64):
            for j in range(64):
                if roImg[i][j] == NoDrive[i][j]:
                    noDrive_val+=1
                if roImg[i][j] == pedestrain[i][j]:
                    pedestrain_val+=1
        print(noDrive_val, " ^   ", pedestrain_val)
        if pedestrain_val>2500:
            print("pedestrain")
        elif noDrive_val>3000:
            print("NoDrive")
        else:
            print("nothing")
    if cv.waitKey(1) == ord("q"):
        break
cap.release()
cv.destroyAllWindows