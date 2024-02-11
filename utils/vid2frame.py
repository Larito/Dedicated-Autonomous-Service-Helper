import numpy as np
import os
import cv2

data_path = "/Users/lara/Desktop/DASH/data/Archive/videos"
saved_path= "/Users/lara/Desktop/DASH/data/Archive/save"
video_paths = os.listdir(data_path)

for video_name in video_paths:
    video = cv2.VideoCapture(os.path.join(data_path, video_name))

    print("Processing", video_name)

    frame_count = 0
    read_next_frame = True
    frame = None
    while True:
        if frame is None or read_next_frame: 
            ret, frame = video.read()
        if not ret: break

        cv2.imshow("frame", frame)
        
        key = cv2.waitKey(0)

        if key == ord("q"): exit()
        if key == ord("k"): break
        if key == ord("s"): 
            frame_count += 1
            vid_name=video_name[:-4]
            file_path = os.path.join(saved_path, f"frame{frame_count}_{vid_name}.jpg")
            cv2.imwrite(file_path, frame)

        if key == ord("r"):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            read_next_frame = False
        else: read_next_frame = True



