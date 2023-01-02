import cv2 # import the opencv library, pip install opencv-python
from time import sleep # import sleep function from time module
from matplotlib import pyplot as plt # import pyplot from matplotlib
import numpy as np # import numpy

video_file = "videos/video3.mp4"
video = cv2.VideoCapture(video_file)

frame_1 = video.read()[1]
frame_2 = video.read()[1]
frame_3 = video.read()[1]

while True:
    frame_1 = frame_2
    frame_2 = frame_3
    ret, frame_3 = video.read()

    key = cv2.waitKey(1)
    if not ret or frame_3 is None or key == ord("q"):
        break

    frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    # frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    frame_3_gray = cv2.cvtColor(frame_3, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(frame_3_gray, frame_1_gray)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    ret, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(thresh)
    b[b == 255] = 0
    g[g == 255] = 0
    thresh = cv2.merge((b, g, r))
    res = cv2.add(thresh, frame_1)

    cv2.imshow("motion_detector", res)
    sleep(1/video.get(cv2.CAP_PROP_FPS))
