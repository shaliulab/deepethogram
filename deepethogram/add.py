import os.path
import yaml
import shutil
import sys
from deepethogram.projects import add_video_to_project


def main():

    args = sys.argv
    
    if len(args) == 3:
        _, project_path, video_path = args
        new_name = None

    elif len(args) == 4:
        _, project_path, video_path, new_name = args
        
    else:
        print("Usage:")
        print("add.py <project_path> <video_path> [new_name]")
        return
        

    with open(os.path.join(project_path, "project_config.yaml"), "r") as filehandle:
        cfg=yaml.load(filehandle, yaml.SafeLoader)
        
    files = os.listdir(project_path)
    assert 'DATA' in files, 'DATA directory not found! {}'.format(files)
    assert 'models' in files, 'models directory not found! {}'.format(files)
    assert 'project_config.yaml' in files, 'project config not found! {}'.format(files)
    assert os.path.exists(video_path), 'video not found {}'.format(video_path)

    if new_name is not None:
        assert os.path.splitext(new_name)[1] in [".mp4", ".avi"]
        # video_path_copied = os.path.join(project_path, "DATA", os.path.splitext(new_name)[0], new_name)
        # os.makedirs(os.path.dirname(video_path_copied))
        # shutil.copyfile(video_path, video_path_copied)
    else:
        new_name = os.path.basename(video_path)        

    transferred_file = add_video_to_project(project=cfg, path_to_video=video_path, mode='copy', basename=new_name)
    print(transferred_file)



if __name__ == "__main__":
    main()