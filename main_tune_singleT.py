import torch
import torch.backends.cudnn as cudnn
from src import *
from src.train_tune_singleT import Optuna_optimizer
import argparse
import os
import sys
import pathlib
import yaml
import logging
import sys
import optuna

sys.dont_write_bytecode = True
cudnn.benchmark = True


def main(args):

    path_to_config = pathlib.Path(args.path)
    
    with open(path_to_config) as f:
        config = yaml.safe_load(f)
        
    #config['data_folder']=pathlib.Path(args.path_data)
    config['model_type'] = args.model
    config['endpoint'] = args.endpoint
    config['output_dir'] = os.path.join(pathlib.Path(config['output_dir']), pathlib.Path(config['endpoint']))
    experiment_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['output_dir']), pathlib.Path(config['model_type']))
    checkpoint_dir = os.path.join(pathlib.Path(experiment_folder), pathlib.Path(config['checkpoint_dir']))
    eval_dir = os.path.join(pathlib.Path(experiment_folder), pathlib.Path(config['eval_dir']))
    
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    model = Optuna_optimizer(config) 
    
    study_name = config['input_image']+"_"+args.model+"_trial"  # Unique identifier of the study.
    storage_name = "sqlite:///db.alessiaDB_{}".format(study_name) #"sqlite:///{}.db"
    
    if args.status == "train":
      
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))   
        study = optuna.create_study(direction = "maximize",study_name=study_name, storage=storage_name, load_if_exists = False) #['maximize', 'maximize']) #, 'maximize', 'maximize'])
        study.enqueue_trial({"batch_size" : 2,
                             "optimizer" : "Adam",
                             "lr" : 0.0001,
                             "weight_decay" : 0.001,
                             "T_max" : 300})
        
    elif args.status == "resume":
        
        study = optuna.create_study(direction = "maximize",study_name=study_name, storage=storage_name, load_if_exists=True)
        
    study.optimize(model.objective_cv, n_trials=100)
    
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print(f"Best trial: {study.best_trial.values}")
    print(f"Best hyperparameters: {study.best_trial.params}")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    parser.add_argument("-e", "--endpoint", type=str, required=True, help="OS/DFS")
    parser.add_argument("-m", "--model", type=str, required=True, help="DenseNet121/ResNet18/SEResNet50")
    parser.add_argument("-s", "--status", type=str, default = "train", help="resume/train")
    args = parser.parse_args()
    
    from monai.config import print_config
    print_config()
    
    main(args)
