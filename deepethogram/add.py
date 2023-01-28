import os.path
import yaml
import shutil
import math
import sys
import argparse
import joblib
from deepethogram.projects import add_video_to_project


SUPPORTED_VIDEO_FORMATS=[".mp4", ".avi"]

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--project-path")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--video-path")
    group.add_argument("--videos-dir")
    ap.add_argument("--chunks", type=int, nargs="+")    
    ap.add_argument("--mode", default="copy", choices=["copy", "symlink", "move"] )
    ap.add_argument("--new-name")
    ap.add_argument("--data-dir", default="DATA")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--n-jobs", default=1)


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
    model_path = args.model_path

     
    if video_path is not None and videos_dir is None:
        add_video(project_path, video_path, data_dir, model_path, new_name, mode)
    else:
        assert chunks is not None
        add_video_chunks_parallel(project_path, videos_dir, data_dir, model_path, chunks=chunks, mode=mode, n_jobs=args.n_jobs)


def add_video_chunks_parallel(chunks, *args, n_jobs=1, **kwargs):



    partition_size = math.ceil(len(chunks) / n_jobs)
    chunk_partition = [chunks[(i*partition_size):((i+1)*partition_size)] for i in range(n_jobs)]

    Output = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(
        add_video_chunks
    )(
        *args, **kwargs, chunks=chunk_partition[i]
    )
        for i in range(len(chunk_partition))
     
    )

    return Output



def build_new_name(video_path, chunk):
    flyhostel = video_path.split(os.path.sep)[-6]
    X = video_path.split(os.path.sep)[-5]
    date_time = video_path.split(os.path.sep)[-4]
    chunk_padded = str(chunk).zfill(6)
    return "_".join([flyhostel, X, date_time, chunk_padded])


def add_video_chunks(project_path, videos_dir, data_dir, model_path, chunks, mode):


    for chunk in chunks:

        video_path = os.path.join(videos_dir, str(chunk).zfill(6) + ".mp4")
        new_name=build_new_name(video_path, chunk)
        add_video(project_path, video_path, data_dir, model_path, new_name, mode=mode)


def add_video(project_path, video_path, data_dir, model_path, new_name, mode="copy"):

    with open(os.path.join(project_path, "project_config.yaml"), "r") as filehandle:
        cfg=yaml.load(filehandle, yaml.SafeLoader)

    
    cfg["project"]["data_path"] = os.path.join(project_path, data_dir)
    cfg["project"]["model_path"] = os.path.join(project_path, model_path)

        
    files = os.listdir(project_path)
    assert data_dir in files, 'DATA directory not found! {}'.format(files)
    assert model_path in files, 'models directory not found! {}'.format(files)
    assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)
    assert os.path.exists(video_path), 'video not found {}'.format(video_path)

    if new_name is not None:
        assert os.path.splitext(new_name)[1] in SUPPORTED_VIDEO_FORMATS
        # video_path_copied = os.path.join(project_path, "DATA", os.path.splitext(new_name)[0], new_name)
        # os.makedirs(os.path.dirname(video_path_copied))
        # shutil.copyfile(video_path, video_path_copied)
    else:
        new_name = os.path.basename(video_path)        

    transferred_file = add_video_to_project(project=cfg, path_to_video=video_path, mode=mode, basename=new_name)
    print(transferred_file)



if __name__ == "__main__":
    main()
