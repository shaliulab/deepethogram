"""
normalize_flow.py


Dissect the first deepethogram step (flow generation) to understand what is it doing with the input videos
The script generates a movie which refines the displayed flows in a movie by normalizing them so they are more clear
to the human eye

Call like so:
    python normalize_flow.py --movie "/root/DATA/foo/foo_flows.mp4"

Or from jupyter, like so:

    from check_flow import normalize_movie
    normalize_movie(movie="/root/DATA/foo/foo_flows.mp4")

The movie is saved to "/root/DATA/foo/foo_flows_norm.mp4"
"""

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
    
    normalize_movie(movie=args.flow)


def normalize_movie(movie):

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


if __name__ == "__main__":
    main()