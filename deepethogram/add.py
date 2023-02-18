import os.path
import re
import shutil
import math
import sys
import argparse
import glob

import joblib
import yaml

from deepethogram.projects import add_video_to_project
from vidio.read import OpenCVReader

ROIS_FILENAME=OpenCVReader.ROIS_FILENAME

SUPPORTED_VIDEO_FORMATS=[".mp4", ".avi"]

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--project-path")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--video-path", help="Path to input video prior to addition to deepethogram project")
    group.add_argument("--videos-dir", help="Folder containing input videos prior to addition to deepethogram project")
    ap.add_argument("--chunks", type=int, nargs="+")
    ap.add_argument("--mode", default="copy", choices=["copy", "symlink", "move"] )
    ap.add_argument("--new-name", type=str, help="""
        New name of the input video. Ignored if video-path is not passed and videos-dir is passed
        """
    )
    ap.add_argument("--data-dir", default="DATA", help="Directory in the deepethogram project where the videos should be linked to")
    ap.add_argument("--n-jobs", default=1, type=int, help="Number of videos linked in parallel, if chunks provided")
    ap.add_argument("--width", default=200, type=int, help="ROI width")
    ap.add_argument("--height", default=200, type=int, help="ROI height")
    ap.add_argument("--stride", default=10, type=int, help="""
    Every stride number of frames is used to compute the video stats.
    The higher this number, the quicker the stats computation. But if it's too high, the stats may be noisier
    """
    )


    return ap

def main():


    ap = get_parser()
    args = ap.parse_args()

    mode = args.mode
    project_path = args.project_path
    video_path = args.video_path
    new_name = args.new_name
    data_dir = args.data_dir
    videos_dir = args.videos_dir
    chunks = args.chunks

    stride=args.stride
    with open(os.path.join(args.project_path, "project_config.yaml"), "r") as filehandle:
        cfg=yaml.load(filehandle, yaml.SafeLoader)


    if video_path is not None and videos_dir is None:
        add_video(
            cfg, project_path, video_path, data_dir, new_name, mode,
            stride=stride, width=args.width, height=args.height
        )
    else:
        assert chunks is not None
        add_video_chunks_parallel(
            chunks, cfg, project_path, videos_dir, data_dir,
            mode=mode, stride=stride, n_jobs=args.n_jobs,
            width=args.width, height=args.height)


def add_video_chunks_parallel(chunks, *args, n_jobs=1, **kwargs):

    partition_size = math.ceil(len(chunks) / n_jobs)
    chunk_partition_ = [chunks[(i*partition_size):((i+1)*partition_size)] for i in range(n_jobs)]
    chunk_partition = []

    for block in chunk_partition_:
        if block:
            chunk_partition.append(block)

    Output = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(
        add_video_chunks
    )(
        *args, **kwargs, chunks=chunk_partition[i]
    )
        for i in range(len(chunk_partition))

    )

    return Output



def build_new_key(video_path, chunk, identity=1):
    flyhostel = video_path.split(os.path.sep)[-6]
    X = video_path.split(os.path.sep)[-5]
    date_time = video_path.split(os.path.sep)[-4]
    chunk_padded = str(chunk).zfill(6)
    identity_padded = str(identity).zfill(3)
    return "_".join([flyhostel, X, date_time, chunk_padded, identity_padded])



def write_rois_file(key_folder, identity, width, height):

    identifier = identity-1
    rois_txt = os.path.join(key_folder, ROIS_FILENAME)
    y = 0

    with open(rois_txt, "w", encoding="utf8") as filehandle:
        x = width * identifier
        filehandle.write(f"{identity} {x} {y} {width} {height}\n")

    return rois_txt

def add_video_chunks(cfg, project_path, videos_dir, data_dir, chunks, width, height, **kwargs):

    number_of_animals = int(re.search("/([0-9])X/", videos_dir).group(1))
    y=0
    EXTENSION=".mp4"


    for chunk in chunks:
        for identity in range(1, number_of_animals+1):

            print(chunk, identity)

            video_path = os.path.join(videos_dir, str(chunk).zfill(6) + EXTENSION)
            key=build_new_key(video_path, chunk, identity)
            key_folder = os.path.join(project_path, data_dir, key)
            os.makedirs(key_folder, exist_ok=True)
            rois_txt=write_rois_file(key_folder, identity, width, height)
            assert os.path.exists(rois_txt)

            try:
                add_video(cfg, project_path, video_path, data_dir, key + EXTENSION, identity, **kwargs)
            except Exception as error:
                print(f"Could not add video for chunk {chunk}")
                print(error)


def add_video(cfg, project_path, video_path, data_dir, new_name, identity=1, mode="copy", stride=10, overwrite=False):
    """
    Add a single fly video to the deepethogram project DATA folder for analysis

    Args:

        project_path (str): Path to folder containing project_config.yaml
        video_path (str): Path to input video
        data_dir (str): Name of folder in deepethogram project where videos should be linked to
        new_name (str): New name of video (with extension)
        identity (int): ROI of the video
        mode (str): One of copy, move or symlink, describing how the video is added to the deepethogram project
        stride (int): Every stride number of frames will be used to compute the stats
    """
    if not overwrite and os.path.exists(video_path) and os.path.exists(os.path.join(os.path.dirname(video_path), "stats.yaml")):
       print(f"{video_path} already symlinked")
       return

    # cfg["project"]["data_path"] = os.path.join(project_path, data_dir)


    files = os.listdir(project_path)
    assert data_dir in files, '{} directory not found! {}'.format(data_dir, files)
    assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)
    assert os.path.exists(video_path), 'video not found {}'.format(video_path)

    if new_name is not None:
        assert os.path.splitext(new_name)[1] in SUPPORTED_VIDEO_FORMATS
        # video_path_copied = os.path.join(project_path, "DATA", os.path.splitext(new_name)[0], new_name)
        # os.makedirs(os.path.dirname(video_path_copied))
        # shutil.copyfile(video_path, video_path_copied)
    else:
        new_name = os.path.basename(video_path)

    data_dir = os.path.join(project_path, data_dir)
    transferred_file = add_video_to_project(project=cfg, path_to_video=video_path, data_path=data_dir, mode=mode, basename=new_name, identity=identity, stride=stride)
    print(f"DONE {transferred_file} -> {data_dir} (stride = {stride})")
    # if mode == "symlink":
    #     dest_file = add_label_to_project(project=cfg, path_to_video=video_path, new_name=new_name, labels_folder=data_dir)
    #     if dest_file is not None: print(f"Linked {dest_file}")



def add_label_to_project(project, path_to_video, new_name, labels_folder):
    """

    """
    data_path = project["project"]["data_path"]

    pattern = os.path.join(labels_folder, os.path.splitext(new_name)[0], "*_labels.csv")
    label_file = glob.glob(pattern)
    key = os.path.splitext(new_name)[0]
    if len(label_file) == 1:
        label_file = label_file[0]
        dest_file = os.path.join(data_path, key,  os.path.basename(label_file))
        if not os.path.exists(dest_file):
            os.symlink(label_file, dest_file)
            return dest_file

    elif len(label_file) == 0:
        print(f"No label file found for {path_to_video} (pattern {pattern})")
    else:
        print(f"Multiple label files found for {path_to_video}")


if __name__ == "__main__":
    main()
