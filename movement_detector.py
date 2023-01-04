import cv2 # import the opencv library, pip install opencv-python
from time import sleep # import sleep function from time module
import numpy as np # import numpy
import os # import os
import shutil # import shutil

def create_entry_folder():
  if "entry_frames" not in os.listdir():
    os.mkdir("entry_frames")
  else:
    shutil.rmtree("entry_frames")
    os.mkdir("entry_frames")


def create_motion_folder():
  if "motion_frames" not in os.listdir():
    os.mkdir("motion_frames")
  else:
    shutil.rmtree("motion_frames")
    os.mkdir("motion_frames")


def create_exit_folder():
  if "exit_frames" not in os.listdir():
    os.mkdir("exit_frames")
  else:
    shutil.rmtree("exit_frames")
    os.mkdir("exit_frames")


def compare_frame_histograms(frame_1, frame_2):
    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    hist_1 = cv2.calcHist([frame_1], [0], None, [256], [0, 256])
    hist_2 = cv2.calcHist([frame_2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)


def detect_movement_area(frame_1, frame_2):
    frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    frame_1_gray = cv2.GaussianBlur(frame_1_gray, (7, 7), 0)
    frame_2_gray = cv2.GaussianBlur(frame_2_gray, (7, 7), 0)

    diff = cv2.absdiff(frame_1_gray, frame_2_gray)
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if compare_frame_histograms(frame_1, frame_2) <= 0.99999:
        b, g, r = cv2.split(thresh)
        b[b == 255] = 0
        g[g == 255] = 0
        motion = cv2.merge((b, g, r))
        return motion
    else:
        return np.zeros_like(frame_1)


if __name__ == "__main__":
    create_entry_folder()
    create_motion_folder()
    create_exit_folder()

    video_file = "videos/video6.mp4"
    video = cv2.VideoCapture(video_file)
    
    while True:
        if video.get(cv2.CAP_PROP_POS_FRAMES) == 0:
            frame_1 = video.read()[1]
            ret, frame_2 = video.read()
        else:
            cv2.imwrite(f"entry_frames/{video.get(cv2.CAP_PROP_POS_FRAMES)}.jpg", frame_2)
            frame_1 = frame_2
            ret, frame_2 = video.read()

        key = cv2.waitKey(1)
        if not ret or key == ord("q"):
            break

        motion = detect_movement_area(frame_1, frame_2)

        res = cv2.add(motion, frame_2)
        cv2.imwrite(f"motion_frames/{video.get(cv2.CAP_PROP_POS_FRAMES)}.jpg", motion)
        cv2.imwrite(f"exit_frames/{video.get(cv2.CAP_PROP_POS_FRAMES)}.jpg", res)
        res = cv2.resize(res, (int(res.shape[1]/2), int(res.shape[0]/2)))
        cv2.imshow("motion_detector", res)
        sleep(1/video.get(cv2.CAP_PROP_FPS))
    cv2.destroyAllWindows()
