import numpy as np
import os
import cv2

data_path = "/Users/laraalotaibi/Desktop/Archive/videos"
saved_path= "/Users/laraalotaibi/Desktop/Archive/save"
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

        height, width = frame.shape[:2]

        height_factor = 1.5
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst_points = np.float32([[0, 0], [width, 0], [width, int(height * height_factor)], [0, int(height * height_factor)]])

        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(frame, perspective_matrix, (width, int(height * height_factor)))

        cv2.imshow("frame", frame)
        cv2.imshow("warped", warped)
        key = cv2.waitKey(0)

        if key == ord("q"): exit()
        if key == ord("k"): break
        if key == ord("s"): 
            frame_count += 1
            cv2.imwrite(saved_path, f"frame{frame_count}.jpg", frame)
            cv2.imwrite(saved_path, f"frame{frame_count}_rotated.jpg", warped)
            read_next_frame = False

        if key == ord("r"):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            read_next_frame = False
        else: read_next_frame = True



