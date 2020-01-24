import imutils
import cv2 

# make stream web camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 24) # Frame frequency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  # The frame width in the video stream.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # The height of the frames in the video stream.

while True:
    # load the image, convert it to grayscale, blur it slightly,
    # and threshold it
    reg, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:	
        if cv2.contourArea(c) > 0:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(image, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the image
    cv2.imshow("finding center", image)
    ch = cv2.waitKey(5)
    if ch == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()    