import monai
import os
import pathlib
import torch
import torch.optim as optim
from src.resnet_blocks import *
from src.ops import BasicBlock, get_inplanes, ResNet
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import nn
from pytorchsummary import summary
from src.models import ConvLSTM_class, ConvLSTM_class2, MSResNet, MSResNet_smaller, DensNet_selfAtt
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, decollate_batch, CacheDataset
from data_preprocessing.load_data import *
import torch.multiprocessing as mp
import optuna
from optuna.trial import TrialState
from src.resnet_blocks import *

seed_everything()

pin_memory = torch.cuda.is_available()


class Optuna_optimizer(object):

    def __init__(self, config):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_points = config['time_steps']

        #####################------------------
        #directories:
        split_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['split_folder']))
        self.experiment_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['output_dir']))

        self.data_folder = pathlib.Path(config['data_folder'])
        self.checkpoint_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['checkpoint_dir']))
        
        self.eval_dir = os.path.join(pathlib.Path(self.experiment_folder), pathlib.Path(config['eval_dir']))
        
        self.output_dir = pathlib.Path(self.experiment_folder)
        
        #####################------------------
        #data loading and pre-processing
        self.outcomes = config['outcomes']
        self.endpoint = config['endpoint']
        self.model_type = config['model_type'] 
        self.time_points = config['time_steps']
        self.input_c_dim = config['input_nc']
        self.image_size = int(config['image_size'])
        
        split_file = os.path.join(split_folder,"data_split_seqNum_"+str(self.time_points)+".txt")
        split_data = read_split_file(split_file)

        self.train_list, self.labels = dataset_prep(split_data['train'],self.data_folder, self.time_points, self.outcomes, self.endpoint)
        self.test_list, _ = dataset_prep(split_data['test'],self.data_folder, self.time_points, self.outcomes, self.endpoint)
        
        images_keys = (config["input_image"],)
        
        self.train_tranforms = initialize_transform(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm",test=False, augmentation = True)
        self.val_tranforms = initialize_transform(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm",test=False, augmentation = False)
        self.test_tranforms = initialize_transform(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm", test=True, augmentation = False)
        
        #####################------------------
        #training settings:

        self.epoch = config['epoch']
        
        if self.model_type=="DenseNet121":
            self.model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=self.input_c_dim, out_channels=2).to(self.device)
        elif self.model_type=="ResNet18":
            self.model = ResNet(block=BasicBlock, layers = [2, 2, 2, 2], block_inplanes = get_inplanes(), n_input_channels=self.input_c_dim, n_classes =2).to(self.device)
        elif self.model_type=="SEResNet50":
            self.model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=self.input_c_dim, num_classes=2).to(self.device)
        else:
            assert False, 'Unknown model type!'
            
        self.loss_function = torch.nn.CrossEntropyLoss() 

        self.num_workers = 1
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        


    def objective(self, trial, train_loader, validate_loader, fold):
        
        self.model.apply(reset_weights)
        self.model.apply(initialize_weights)
        
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        
        optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        T_max = trial.suggest_int("T_max", 100, 400, step = 50)
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
        
        path_save_checkpoint = self.checkpoint_dir+f"/earlyStop_checkpoint_{fold}.pt"
        
        early_stopping = EarlyStopping(patience=50, verbose=True, path=None)
        
        for epoch in range(self.epoch):
            
            self.model.train()
            
            step = 0
            labT = []
            predT = []
            
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["input"].to(self.device), batch_data["label"].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                labels = torch.argmax(labels, dim=1)
                
                labT.extend(labels.cpu().numpy())
                predT.extend(outputs.argmax(dim=1).cpu().numpy())
                        
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
                
            self.model.eval()
            
            count = 0
            num_correct = 0.0
            metric_count = 0
                
            lab = []
            pred = []
                
            loss_val = 0
            
            for val_data in validate_loader:
                
                val_images, val_labels = val_data["input"].to(self.device), val_data["label"].to(self.device)
                val_labels = torch.argmax(val_labels, dim=1)
                
                with torch.no_grad():
                    val_outputs = self.model(val_images)
                    
                    lossV = self.loss_function(val_outputs, val_labels)
                    loss_val += lossV.item()
                    
                    predicted_labels = val_outputs.argmax(dim=1)
                    
                    num_correct += (predicted_labels == val_labels).sum().item()
                    metric_count += val_labels.size(0)
                    
                    lab.extend(val_labels.cpu().numpy())
                    pred.extend(predicted_labels.cpu().numpy())
                    
                    count += 1
                
            torch.cuda.empty_cache()
            
            f1 = f1_score(np.array(lab), np.array(pred), average='macro')
               
            auc = roc_auc_score(np.array(lab), np.array(pred), multi_class='ovr')
               
            metric = num_correct / metric_count
            
            composite_metric = (0.4 * f1) + (0.4 * auc) + (0.2 * metric)
            trial.report(composite_metric, epoch)
            
            trial.set_user_attr(f"accuracy", metric)
            trial.set_user_attr(f"auc", auc)
            trial.set_user_attr(f"f1_score", f1)
            value_loss = loss_val/ count
            trial.set_user_attr(f"val_loss", value_loss)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            early_stopping(value_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        return composite_metric
        
       
    def objective_cv(self, trial):
        
        kfolds = 3
        batch_size = trial.suggest_int("batch_size", 1, 5)
        
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
        
        for fold, (train_index, val_index) in enumerate(kf.split(X = self.train_list, y=np.array(self.labels))):
            if fold == 0:
                train_data = [self.train_list[i] for i in train_index]
                train_labels = [self.labels[i] for i in train_index]
                
                train_ds = Dataset(data=train_data, transform=self.train_tranforms)
                sample_weights = calculate_sample_weights(train_labels)
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
                train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=self.num_workers, pin_memory=False) #changed to False
            
                val_data = [self.train_list[i] for i in val_index]
                validate_ds = Dataset(data=val_data, transform=self.val_tranforms)
                validate_loader = DataLoader(validate_ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False) #changed to False

                try:
                    composite_metric = self.objective(trial, train_loader, validate_loader, fold+1)
                    
                except optuna.exceptions.TrialPruned:
                    print(f"Trial pruned at fold {fold+1}")
                    return float('nan')
        
        return np.mean(composite_metric) if composite_metric else float('nan')
        

