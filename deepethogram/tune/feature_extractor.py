import os
import sys

from omegaconf import OmegaConf, DictConfig
try: 
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch
except ImportError:
    print('To use the deepethogram.tune module, you must `pip install \'ray[tune]`')
    raise

from deepethogram.configuration import make_config, load_config_by_name
from deepethogram import feature_extractor_train
from deepethogram import projects
from deepethogram.tune.utils import dict_to_dotlist, generate_tune_cfg

def tune_feature_extractor(cfg: DictConfig):    
    # tune_cfg = {
    #     'feature_extractor.dropout_p': tune.uniform(0.0, 0.9), 
    #     'train.regularization.alpha': tune.uniform(1e-5, 0.01), 
    #     'train.regularization.beta': tune.uniform(1e-6, 1e-3), 
    #     'train.loss_gamma': tune.choice([0, 0.5, 1, 2, 5]), 
    #     'train.loss_weight_exp': tune.uniform(0.0, 1.0)
    # }
    
    # if cfg is None:
    #     cfg = load_config_by_name('tune')
    
    scheduler = ASHAScheduler(
        max_t=cfg.train.num_epochs, # epochs
        grace_period=cfg.tune.grace_period,
        reduction_factor=2)
    
    reporter_dict = {}
    for key in cfg.tune.hparams.keys():
        reporter_dict[key] = cfg.tune.hparams[key].short
    # reporter_dict = {key: value for key, value in zip(cfg.tune.hparams.keys(), )}
    reporter = CLIReporter(parameter_columns=reporter_dict)
    
    # this converts what's in our cfg to a dictionary containing the search space of our hyperparameters
    tune_experiment_cfg = generate_tune_cfg(cfg)
    
    if cfg.tune.search == 'hyperopt':
        # https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-hyperopt
        current_best = {}
        for key, value in cfg.tune.hparams.items():
            current_best[key] = value.current_best
        # hyperopt wants this to be a list of dicts
        current_best = [current_best]
        search = HyperOptSearch(metric=cfg.tune.key_metric, 
                                mode='max', 
                                points_to_evaluate=current_best)
    elif cfg.tune.search == 'random':
        search = None
    else: 
        raise NotImplementedError
    
    print('Running hyperparamter tuning with configuration: ')
    print(OmegaConf.to_yaml(cfg))
    
    analysis = tune.run(
        tune.with_parameters(
            run_ray_experiment, 
            cfg=cfg,
        ), 
        resources_per_trial=OmegaConf.to_container(cfg.tune.resources_per_trial), 
        metric=cfg.tune.key_metric, 
        mode='max', 
        config=tune_experiment_cfg,
        num_samples=cfg.tune.num_trials, # how many experiments to run
        scheduler=scheduler, 
        progress_reporter=reporter, 
        name=cfg.tune.name, 
        local_dir=cfg.project.model_path, 
        search_alg=search
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    analysis.results_df.to_csv(os.path.join(cfg.project.model_path, 'ray_results.csv'))


def run_ray_experiment(ray_cfg, cfg): 
    # cfg = make_feature_extractor_train_cfg(project_path, use_command_line=False, preset='deg_f')
    # tune_cfg = load_config_by_name('tune')
    
    ray_cfg = OmegaConf.from_dotlist(dict_to_dotlist(ray_cfg))
    
    cfg = OmegaConf.merge(cfg, ray_cfg)
    # cfg.tune.use = True
    
    # cfg.flow_generator.weights = 'latest'
    # cfg.feature_extractor.weights = '/media/jim/DATA_SSD/niv_revision_deepethogram/models/pretrained_models/200415_125824_hidden_two_stream_kinetics_degf/checkpoint.pt'
    # cfg.compute.batch_size = 64
    # cfg.train.steps_per_epoch.train = 20
    # cfg.train.steps_per_epoch.val = 20
    if cfg.notes is None:
        cfg.notes = f'{cfg.tune.name}_{tune.get_trial_id()}'
    else:
        cfg.notes += f'{cfg.tune.name}_{tune.get_trial_id()}'
    feature_extractor_train(cfg)
    
if __name__ == '__main__':
    # USAGE
    # to run locally, type `ray start --head --port 6385`, then run this script
    
    ray.init(address='auto')  #num_gpus=1
    
    config_list = ['config','augs','model/flow_generator','train', 'model/feature_extractor', 'tune']
    run_type = 'train'
    model = 'feature_extractor'
    
    project_path = projects.get_project_path_from_cl(sys.argv)
    cfg = make_config(project_path=project_path, config_list=config_list, run_type=run_type, model=model, 
                      use_command_line=True, debug=True)
    cfg = projects.convert_config_paths_to_absolute(cfg)
    
    if 'preset' in cfg.keys():
        cfg.tune.name += '_{}'.format(cfg.preset)
    if 'debug' in cfg.keys():
        cfg.tune.name += '_debug'
    
    tune_feature_extractor(cfg)
    
    