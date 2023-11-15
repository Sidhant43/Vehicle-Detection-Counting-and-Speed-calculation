
import cv2
import numpy as np


# Open the video file
cap = cv2.VideoCapture('video.mp4')
min_width_rect = 80
min_height_rect = 80
count_line_position = 550

# Initialize the background subtractor
algo = cv2.createBackgroundSubtractorMOG2()


while True:
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    cv2.imshow('Detecter',dilatada)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()