import numpy as np
import cv2

if __name__ == '__main__':
    def nothing(*arg):
        pass

cv2.namedWindow( "result" ) # make main window
cv2.namedWindow( "settings" ) # make window of setting

cap = cv2.VideoCapture(0)

# create 6 sliders to adjust the start and end color of the filter
cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
crange = [0,0,0, 0,0,0]

while True:
    # read "video" (image)
    flag, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
 
    # read the values of the runners
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')

    # form the initial and final color of the filter
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # apply a filter to the frame in the HSV model
    thresh = cv2.inRange(hsv, h_min, h_max)

    # to see the result
    cv2.imshow('result', thresh) 
 
    # exit condition
    ch = cv2.waitKey(5)
    if ch == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


