import torch
import torch.backends.cudnn as cudnn
from src import *
from src.train import ImageSequence_Classification
import argparse
import os
import sys
import pathlib
import yaml
from src.utils import *

seed_everything()

sys.dont_write_bytecode = True
cudnn.benchmark = True

def main(args):

    path_to_config = pathlib.Path(args.path)
    
    with open(path_to_config) as f:
        config = yaml.safe_load(f)
        
    #config['data_folder']=pathlib.Path(args.path_data)
    config['phase'] = args.phase
    config['foldNum'] = args.fold_num
    config['endpoint'] = args.endpoint
    config['output_dir'] = os.path.join(pathlib.Path(config['output_dir']), pathlib.Path(config['endpoint']))
    config['masked_images'] = args.masked
    experiment_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['output_dir']))

    checkpoint_dir = os.path.join(pathlib.Path(experiment_folder), pathlib.Path(config['checkpoint_dir']))
    eval_dir = os.path.join(pathlib.Path(experiment_folder), pathlib.Path(config['eval_dir']))
    
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    #model_type = config['model_type']
    
    #try:
    model = ImageSequence_Classification(config)   
    #except:
    #assert False, 'Unknown model'

    #if config['phase'] == 'optim':
        #model.optimize(config)
    if config['phase'] == 'train':
        model.train(config)
    elif config['phase'] == 'test':
        model.test(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    parser.add_argument("-e", "--endpoint", type=str, required=True, help="OS/DFS")
    parser.add_argument("-ph", "--phase", type=str, required=True, help="train/test")
    parser.add_argument("-mk", "--masked", default=False, help="images masked or not")
    parser.add_argument("-f", "--fold_num", default=-1, type=int, help="fold number - 1")
    args = parser.parse_args()
    
    from monai.config import print_config
    print_config()
    
    main(args)
