import os
import logging
import argparse
import h5py
from omegaconf import DictConfig, OmegaConf
import cv2

from deepethogram import utils, viz, projects

DEEPETHOGRAM_PROJECT_PATH=os.environ["DEEPETHOGRAM_PROJECT_PATH"]
log = logging.getLogger(__name__)

from deepethogram.configuration import make_feature_extractor_inference_cfg, make_sequence_inference_cfg
from deepethogram.data.augs import get_cpu_transforms, get_gpu_transforms_inference, get_gpu_transforms
from deepethogram.flow_generator.train import build_model_from_cfg as build_flow_generator
from deepethogram.flow_generator.inference import extract_movie
from deepethogram.feature_extractor.train import build_model_from_cfg as build_feature_extractor
from deepethogram.sequence.train import build_model_from_cfg as build_sequence
cwd=os.getcwd()
import shutil


CONFIG_MAKERS={
    "feature_extractor": make_feature_extractor_inference_cfg,
    "sequence": make_sequence_inference_cfg,
}
BUILDERS={
    "flow_generator": build_flow_generator,
    "feature_extractor": lambda x: build_feature_extractor(x)[4],
    "sequence": build_sequence,
}


def add_weights_to_cfg(step, cfg, weights):
    """
    Bind some weights to the config

    """
    try:
        run_files = utils.get_run_files_from_weights(weights)
        loaded_config_file = run_files['config_file']
        loaded_cfg = OmegaConf.load(loaded_config_file)
        loaded_model_cfg = getattr(loaded_cfg, step)
        current_model_cfg = getattr(cfg, step)
        model_cfg = OmegaConf.merge(current_model_cfg, loaded_model_cfg)
        setattr(cfg, step, model_cfg)
        cfg.metrics_file=run_files["metrics_file"]
    except:
        pass
    
    # we don't want to use the weights that the trained model was initialized with
    # Instead, we want the weights after training
    # therefore, overwrite the loaded configuration with the current weights
    setattr(getattr(cfg, step), "weights", weights)
    return cfg
    
def load_model(step, tag="latest"):
    """
    Load a DEG model

    For sure works with flow generator models

    Arguments:

        step (str): model type, one of flow_generator, feature_extractor, sequence
        tag (str): either latest or path to the checkpoint file in a model train folder
    """
   
    if step != "flow_generator":
        cfg=CONFIG_MAKERS[step](DEEPETHOGRAM_PROJECT_PATH)
    else:
        cfg=CONFIG_MAKERS["feature_extractor"](DEEPETHOGRAM_PROJECT_PATH)

    cfg = projects.setup_run(cfg)
    
    
    setattr(getattr(cfg, step), "weights", tag)
    if step == "feature_extractor":
        setattr(getattr(cfg, "flow_generator"), "weights", "latest")
    
    weights = projects.get_weightfile_from_cfg(cfg, step)

    assert os.path.isfile(weights)
    
    cfg = add_weights_to_cfg(step, cfg, weights)
    kwargs={}
    if step == "sequence":
        kwargs.update({
            "num_features": 1024,
            "num_classes": len(cfg.project.class_names)
        })
    
    
    model_components = BUILDERS[step](cfg, **kwargs)
    model = model_components
    device = 'cuda:{}'.format(cfg.compute.gpu_id)

    if step != "flow_generator":
        metrics_file = cfg.metrics_file
        assert os.path.isfile(metrics_file)
        best_epoch = utils.get_best_epoch_from_weightfile(weights)
        # best_epoch = -1
        log.info('best epoch from loaded file: {}'.format(best_epoch))
        with h5py.File(metrics_file, 'r') as f:
            try:
                thresholds = f['val']['metrics_by_threshold']['optimum'][best_epoch, :]
            except KeyError:
                # backwards compatibility
                thresholds = f['threshold_curves']['val']['optimum'][best_epoch, :]
                
        if len(thresholds) != len(list(cfg.project.class_names)):
            error_message = '''Number of classes in trained model: {}
                Number of classes in project: {}
                Did you add or remove behaviors after training this model? If so, please retrain!
            '''.format(len(thresholds), len(cfg.project.class_names))
            raise ValueError(error_message)

    else:
        thresholds = None
    return cfg, model, thresholds



def build_out_video_path(movie, movie_format):
    """
    Produce a systematic name for a video containing the result of a flow computation
    """
    out_video = os.path.splitext(movie)[0] + '_flows'
    if movie_format == 'directory':
        pass
    elif movie_format == 'hdf5':
        out_video += '.h5'
    elif movie_format == 'ffmpeg':
        out_video += '.mp4'
    else:
        out_video += '.avi'
    if os.path.isdir(out_video):
        shutil.rmtree(out_video)
    elif os.path.isfile(out_video):
        os.remove(out_video)
    return out_video


def get_parser():

    ap=argparse.ArgumentParser()
    ap.add_argument("--movie")
    ap.add_argument("--tag", default="latest")
    return ap

def evaluate_flow_generator(movie, model="latest"):
    cwd=os.getcwd()
    
    cfg=CONFIG_MAKERS["feature_extractor"](DEEPETHOGRAM_PROJECT_PATH)
    cfg = projects.setup_run(cfg)

    cfg, model, thresholds= load_model("flow_generator", model)
    device = 'cuda:{}'.format(cfg.compute.gpu_id)
    model = model.to(device)
    os.chdir(cwd)


    movie_format = 'ffmpeg'
    maxval = 5
    polar = True
    save_rgb_side_by_side = True
    cpu_transform = get_cpu_transforms(cfg.augs)['val']
    mode = '3d' if '3d' in cfg.feature_extractor.arch.lower() else '2d'
    gpu_transform = get_gpu_transforms(cfg.augs, mode)
    out_video=build_out_video_path(movie, movie_format)
    input_images = 11
    fps=int(cv2.VideoCapture(movie).get(5))


    # the master branch of deepethogram does not let you customise the fps of the output video
    # it is set to 30
    extract_movie(
        movie,
        out_video,
        model,
        device,
        cpu_transform,
        gpu_transform,
        mean_by_channels=cfg.augs.normalization.mean,
        num_workers=1,
        num_rgb=input_images,
        maxval=maxval,
        polar=polar,
        movie_format=movie_format,
        save_rgb_side_by_side=save_rgb_side_by_side,
    )

def main():
    ap = get_parser()
    args = ap.parse_args()
    evaluate_flow_generator(args.movie, args.tag)


if __name__ == "__main__":
    main()