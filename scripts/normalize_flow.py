import argparse
import os.path

import numpy as np
from tqdm.auto import tqdm
import cv2


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True)

def main():

    ap = get_parser()
    args=ap.parse_args()
    
    normalize_flow(movie=args.flow)


def normalize_flow(movie):

    cap=cv2.VideoCapture(movie)
    fps=int(cap.get(5))
    width=int(cap.get(3))
    height=int(cap.get(4))
    file, ext = os.path.splitext(movie)
    norm_flow = file + "_norm" + ext

    video_writer=cv2.VideoWriter(norm_flow, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), isColor=True)
    i=0
    pb=tqdm()
    # absolute_min=255
    # absolute_max=0

    while True:
        ret, frame=cap.read()
        if not ret:
            break
        frame=normalize_frame(frame)
        video_writer.write(frame)
        i+=1
        pb.update(1)
    video_writer.release()
    cap.release()


def normalize_frame(frame):

    original = np.uint8(frame[:, frame.shape[1]//2:, :])
    frame=frame[:, :frame.shape[1]//2,:]
    # absolute_min = min(max(0, frame.min()), absolute_min)
    # absolute_max = max(min(255, frame.max()), absolute_max)
    
    frame=frame-frame.min()
    frame=255*(frame/frame.max())
    frame=np.uint8(frame)
    frame=np.hstack([frame, original])
    return frame