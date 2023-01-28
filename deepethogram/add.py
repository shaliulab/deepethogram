import os.path
import yaml
import shutil
import sys
import argparse
from deepethogram.projects import add_video_to_project


SUPPORTED_VIDEO_FORMATS=".mp4", ".avi"]

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--project-path")
    ap.add_argument("--video_path")
    ap.add_argument("--mode", default="copy", choices=["copy", "symlink", "move"] )
    ap.add_argument("--new-name")
    ap.add_argument("--data-dir", default="DATA")
    ap.add_argument("--model-dir", default="models")
    return ap

def main():


    ap = get_parser()
    args = ap.parse_args()

    mode = args.mode
    project_path = args.project_path
    video_path = args.video_path
    new_name = args.new_name
    data_dir = args.data_dir
    model_path = args.model_path
    

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