"""
loader.py
从data下每个视频中提取
"""
import os
import cv2
import numpy as np


def modify(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channel = 3

    data = np.empty([length, channel, height, width])
    count = 0
    while cap.isOpened() and count < length:
        ret, frame = cap.read()
        if ret:
            data[count, :, :, :] = np.transpose(frame, (2, 0, 1))
            count += 1
    cap.release()
    np.save(save_path, data)


def main():
    for idx in range(1, 10):
        video_dir = os.path.join(os.path.abspath('.'), 'data', 'Video_%03d' % idx)
        modify(
            video_path=os.path.join(video_dir, 'Video_%03d.avi' % idx),
            save_path=os.path.join(video_dir, 'BMC2012_%03d.npy' % idx)
        )


if __name__ == '__main__':
    main()
