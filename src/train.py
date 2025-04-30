import monai
import os
import pathlib
import torch
from src.utils import *
import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import nn
from pytorchsummary import summary
from src.models import ConvLSTM_class, ConvLSTM_class2, MSResNet, MSResNet_smaller, DensNet_selfAtt, ResNet_selfAtt, SEResNet_selfAtt, DensNet_TCN, ResNet_TCN, SEResNet_TCN
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, decollate_batch, CacheDataset
from data_preprocessing.load_data import *
import torch.multiprocessing as mp


pin_memory = torch.cuda.is_available()

#https://github.com/atmguille/Violence-Detection-With-Human-Skeletons/blob/main/experiments/RWF-2000/notebooks/Best%20model%20-%20final%20traning%20(last%2020%20epochs).ipynb

class ImageSequence_Classification(object):

    def __init__(self, config):

        #self.manager = mp.Manager()
        #self.results = self.manager.dict()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        self.time_points = config['time_steps']

        #####################------------------
        #directories:
        self.split_folder = os.path.join(pathlib.Path(config['experiment_dir']), pathlib.Path(config['split_folder']))
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
        self.kernel_size = int(config['kernel_size'])

        split_file = os.path.join(self.split_folder,"data_split_seqNum_"+str(self.time_points)+".txt")
        split_data = read_split_file(split_file)

        if self.time_points == self.input_c_dim:
            planningCT=False
        else:
            planningCT=True
        
        self.train_list, self.labels = dataset_prep(split_data['train'],self.data_folder, self.time_points, self.outcomes, self.endpoint, planningCT)
        self.test_list, _ = dataset_prep(split_data['test'],self.data_folder, self.time_points, self.outcomes, self.endpoint, planningCT)
        
        images_keys = tuple([j for j in self.test_list[0].keys() if "ct" in j])
        print(images_keys)
        
        if config["masked_images"]:
            print("GTV included!")
            self.train_tranforms = initialize_transform_gtv(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm",test=False, augmentation = False)
            self.val_tranforms = initialize_transform_gtv(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm",test=False, augmentation = False)
            self.test_tranforms = initialize_transform_gtv(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm", test=True, augmentation = False)
        else:
            print("no GTV included!")
            self.train_tranforms = initialize_transform(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm",test=False, augmentation = True)
            self.val_tranforms = initialize_transform(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm",test=False, augmentation = False)
            self.test_tranforms = initialize_transform(images_keys, self.model_type, level=60, window=350, image_size = self.image_size, ct_norm="z_norm", test=True, augmentation = False)
        
        self.use_gtv = config["masked_images"]
        
        #####################------------------
        #training settings:

        self.epoch = config['epoch']
        self.batch_size = config['batch_size'] #batch size indicates the number of slices which we are considering in a sequence
        self.lr = config['lr']
        self.save_freq = config['save_freq']
        self.kfolds = config['kfolds']
        self.foldNum = config['foldNum']

        if self.use_gtv:
            chann = 2
        else:
            chann = 1
        
        if self.model_type=="DenseNet121":
            self.model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=self.input_c_dim, out_channels=2).to(self.device)
        elif self.model_type=="ConvLSTM_class":
            self.model = ConvLSTM_class2(spatial_dims=3, in_channels=1, out_channels=2, len_sequence = self.time_points + 1, kernel_size = self.kernel_size, device = self.device).to(self.device)
        elif self.model_type=="MS-ResNet":
            self.model = MSResNet(batch_size = self.batch_size, spatial_dims=3, in_channels=1, out_channels=2, len_sequence = self.time_points + 1, kernel_size = self.kernel_size, device = self.device).to(self.device)
        elif self.model_type=="MS-ResNet_smaller":
            self.model = MSResNet_smaller(batch_size = self.batch_size, spatial_dims=3, in_channels=1, out_channels=2, len_sequence = self.time_points + 1, kernel_size = self.kernel_size, device = self.device).to(self.device)
        elif self.model_type=="DensNet_selfAtt":
            self.model = DensNet_selfAtt(num_classes = 2, in_channels=chann).to(self.device)
        elif self.model_type=="ResNet_selfAtt":
            self.model = ResNet_selfAtt(num_classes = 2, in_channels=chann).to(self.device)
        elif self.model_type=="SEResNet_selfAtt":
            self.model = SEResNet_selfAtt(num_classes = 2, in_channels=chann).to(self.device)
        elif self.model_type=="DensNet_TCN":
            self.model = DensNet_TCN(num_classes = 2, in_channels=chann).to(self.device)
        elif self.model_type=="ResNet_TCN":
            self.model = ResNet_TCN(num_classes = 2, in_channels=chann).to(self.device)
        elif self.model_type=="SEResNet_TCN":
            self.model = SEResNet_TCN(num_classes = 2, in_channels=chann).to(self.device)
        else:
            assert False, 'Unknown model type!'

        if config['optimizer']=="SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif config['optimizer']=="Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=config['weight_decay'])
        elif config['optimizer']=="AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=config['weight_decay'])
        
        if config['scheduler']=="CosineAnnealingWarm":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,  T_0=10, T_mult=2)  #30 failed, 60 did not try
        elif config['scheduler']=="CosineAnnealing":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epoch_step'], eta_min=0)
        else:
            self.lr_scheduler = None
        
        self.loss_type = config['loss_type']
        if self.loss_type == "CrossEntropy":
            self.loss_function = torch.nn.NLLLoss() 
        elif self.loss_type == "CrossEntropy_weighted":
            self.loss_function = torch.nn.NLLLoss(weight=torch.tensor([0.3, 0.7]).to(self.device)) 
        
        self.num_workers = 2 #int(os.environ["SLURM_CPUS_ON_NODE"]) #config['num_workers']
        #self.model = torch.nn.DataParallel(self.model).to(self.device)


    def train_loop(self, fold, train_loader, train_ds, validate_ds, validate_loader):
        
        best_metric = -1
        best_f1 = -1
        best_f1_epoch = -1
        best_metric_epoch = -1
        best_overall_metric = -1
        best_overall_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        f1_values = []
        auc_values = [] 
        f1_values_train = []
        val_loss = []

        writer = SummaryWriter()
        
        self.model.apply(reset_weights)
        self.model.apply(initialize_weights)
        
        early_stopping = EarlyStopping(patience=50, verbose=True, path=self.checkpoint_dir+f"/earlyStop_checkpoint_{fold}.pt")
        
        for epoch in range(self.epoch):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.epoch}")
            
            self.model.train()
            
            epoch_loss = 0
            step = 0
            labT = []
            predT = []

            for batch_data in train_loader:
                step += 1
                
                if self.use_gtv: 
                    input, gtv, labels = batch_data["input"].to(self.device), batch_data["mask"].to(self.device), batch_data["label"].to(self.device)
                    input = input.unsqueeze(1)
                    gtv = gtv.unsqueeze(1)
                    inputs = torch.cat((input,gtv), 1)
                else:
                    inputs, labels = batch_data["input"].to(self.device), batch_data["label"].to(self.device)
                    inputs = inputs.unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                labels = torch.argmax(labels, dim=1)
                
                labT.extend(labels.cpu().numpy())
                predT.extend(outputs.argmax(dim=1).cpu().numpy())
                        
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            
            if self.lr_scheduler != None:
                    self.lr_scheduler.step()
            
            f1_train = f1_score(np.array(labT), np.array(predT), average='macro')
            f1_values_train.append(f1_train)    
            
            torch.cuda.empty_cache()
            
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.save_freq == 0:
                
                self.model.eval()
                count = 0
                val_epoch = 0
                num_correct = 0.0
                metric_count = 0
                
                lab = []
                pred = []
                
                for val_data in validate_loader:
                    
                    if self.use_gtv: 
                        val_image, gtv_val, val_labels = val_data["input"].to(self.device), val_data["mask"].to(self.device), val_data["label"].to(self.device)
                        val_image = val_image.unsqueeze(1)
                        gtv_val = gtv_val.unsqueeze(1)
                        val_images = torch.cat((val_image,gtv_val), 1)     
                    else:
                        val_images, val_labels = val_data["input"].to(self.device), val_data["label"].to(self.device)
                        val_images = val_images.unsqueeze(1)
                        
                    val_labels = torch.argmax(val_labels, dim=1)
                    
                    with torch.no_grad():
                        val_outputs = self.model(val_images)
                        
                        lossV = self.loss_function(val_outputs, val_labels)
                        
                        predicted_labels = val_outputs.argmax(dim=1)
                        
                        num_correct += (predicted_labels == val_labels).sum().item()
                        metric_count += val_labels.size(0)
                        
                        val_epoch += lossV.item()
                        #epoch_len_val = len(validate_ds) // validate_loader.batch_size
                        
                        lab.extend(val_labels.cpu().numpy())
                        pred.extend(predicted_labels.cpu().numpy())
                        
                        count += 1
                        
                torch.cuda.empty_cache()
                
                if count==1:
                    print_image(val_images, path = self.eval_dir)
                
                f1 = f1_score(np.array(lab), np.array(pred), average='macro')
                f1_values.append(f1)
                
                auc = roc_auc_score(np.array(lab), np.array(pred), multi_class='ovr')
                auc_values.append(auc)
                
                metric = num_correct / metric_count
                
                val_loss_avg = val_epoch / len(validate_loader)
                val_loss.append(val_loss_avg)
                
                metric_values.append(metric)
                
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir,"last_model_fold"+str(fold)+".pth"))

                if metric >= best_metric and epoch>15:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    #torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir,"best_metric_model_fold"+str(fold)+".pth"))
                    print("saved new best metric model")

                if f1 >= best_f1 and epoch>15:
                    best_f1 = f1
                    best_f1_epoch = epoch + 1
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir,"best_f1_model_fold"+str(fold)+".pth"))
                
                if (f1+auc+metric)/3 >= best_overall_metric:
                    best_overall_metric = (f1+auc+metric)/3
                    best_overall_metric_epoch = epoch + 1
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir,"best_overall_metric_model_fold"+str(fold)+".pth"))
                    print("saved new best metric model")
                    
                early_stopping(val_loss_avg, self.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} current f1: {f1:.4f}")
                print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}. Best f1: {best_f1:.4f} at epoch {best_f1_epoch}")
                writer.add_scalar("val_accuracy", metric, epoch + 1)
                writer.add_scalar("val_f1", f1, epoch + 1)
                
            name_file = self.eval_dir + '/results_Fold'+str(fold)+'.xlsx'
            results = pd.DataFrame({"Train F1" : f1_values_train, "Val Accuracy": metric_values, "Val F1" : f1_values, "Val AUC": auc_values, "Train Loss": epoch_loss_values, "Val Loss": val_loss})
            results.to_excel(name_file, index=False)
            
            message = f"Best F1 Score checkpoint save at epoch: {best_f1_epoch} with score {best_f1:.4f}. Best overall Score checkpoint save at epoch: {best_overall_metric_epoch} with score {best_overall_metric:.4f}."
        
            with open(self.eval_dir + '/Info_Fold'+str(fold)+'.txt', 'w') as file:
                file.write(message)
        

        print(f"Training completed, best_metric for fold {fold}: {best_metric:.4f} at epoch: {best_metric_epoch}")
        print(" ")
        
        writer.close() 
        
       
    def train(self, config):
        
        kf = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=42)
        
        for fold, (train_index, val_index) in enumerate(kf.split(X = self.train_list, y=np.array(self.labels))):
            if (self.foldNum>=0 and fold == self.foldNum) or self.foldNum<0:
                train_data = [self.train_list[i] for i in train_index]
                train_labels = [self.labels[i] for i in train_index]
                train_ds = Dataset(data=train_data, transform=self.train_tranforms)
                
                sample_weights = calculate_sample_weights(train_labels)
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            
                train_loader = DataLoader(train_ds, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, pin_memory=False) #changed to False
                
                val_data = [self.train_list[i] for i in val_index]
                validate_ds = Dataset(data=val_data, transform=self.val_tranforms)
                validate_loader = DataLoader(validate_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False) #changed to False

                print(f"Training starting for fold {fold+1}")
                print("ratio events in training: " + str(np.sum([self.labels[i] for i in train_index])/len(train_index)))
                print("ratio events in validation: " + str(np.sum([self.labels[i] for i in val_index])/len(val_index)))
                
                self.train_loop(fold+1, train_loader, train_ds, validate_ds, validate_loader)
            else:
                print("Condition not met. Skip this fold.")


    def load(self, checkpoint_dir):

        print(" [*] Reading checkpoint...")
        try:
            self.model.load_state_dict(torch.load(checkpoint_dir))
            return True
        except Exception as e:
            print(e)
            return False

    def test(self, config):

        ds = Dataset(data=self.test_list, transform=self.test_tranforms)
        test_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=pin_memory)
        
        metrics = list()
        all_labels = []
        all_outputs = []
        all_probs = []
        
        for fold in range(1,self.kfolds+1): 
            
            print(f"Processing fold {fold}")
            
            check_ = "best_f1_model"
            
            name_checkpoint = check_ + f"_fold{fold}.pth" #f"earlyStop_checkpoint_{fold}.pt" 
            checkpoint_dir = os.path.join(self.checkpoint_dir,name_checkpoint)
            
            if self.load(checkpoint_dir):
                print(" [*] Load SUCCESS checkpoint " + checkpoint_dir)
            else:
                print(" [!] Load failed of checkpoint " + checkpoint_dir)
                
            self.model.eval()

            labels = list()
            outputs = list()
            probs = list()
            
            with torch.no_grad():
                for test_data in test_loader:
                    
                    test_images, test_labels = test_data["input"].to(self.device), test_data["label"].to(self.device) 
                    test_images = test_images.unsqueeze(1)
                    
                    if self.use_gtv: 
                        
                        test_image, test_gtv, test_labels = test_data["input"].to(self.device), test_data["mask"].to(self.device), test_data["label"].to(self.device) 
                        test_image = test_image.unsqueeze(1)
                        test_gtv = test_gtv.unsqueeze(1)
                        test_image = torch.cat((test_image,test_gtv), 1) 
                    else:
                        test_images, test_labels = test_data["input"].to(self.device), test_data["label"].to(self.device) 
                        test_images = test_images.unsqueeze(1)
                    
                    test_outputs = self.model(test_images)
                    
                    probabilities = torch.nn.functional.softmax(test_outputs, dim=1)
                    
                    labels.append(test_labels.argmax(dim=1).cpu().numpy())
                    outputs.append(test_outputs.argmax(dim=1).cpu().numpy())
                    probs.append(probabilities.cpu().numpy()[0][-1])
                    
            labels = np.concatenate(labels)
            outputs = np.concatenate(outputs)
            
            all_labels.append(labels)
            all_outputs.append(outputs)
            all_probs.append(probs)
            
            fold_metrics = calculate_metrics(labels, outputs)
            fold_metrics['Fold'] = fold
            metrics.append(fold_metrics)
            
        all_labels_flat = np.array([label for sublist in all_labels for label in sublist])
        all_outputs_flat = np.array([output for sublist in all_outputs for output in sublist])
        all_probs_flat = np.array([prob for sublist in all_probs for prob in sublist])  # Flatten probabilities

        num_samples = len(all_labels[0])
        
        majority_labels = all_labels_flat.reshape(self.kfolds, num_samples).T
        majority_outputs = all_outputs_flat.reshape(self.kfolds, num_samples).T 
        majority_probs = all_probs_flat.reshape(self.kfolds, num_samples)#.T
        majority_probs_T = majority_probs.T
        
        final_labels = []
        final_outputs = []
        prob = []
        final_avg_output = []

        for i in range(majority_labels.shape[0]):
            final_labels.append(majority_labels[i, 0])  
            final_outputs.append(np.bincount(majority_outputs[i]).argmax())  
            avg = np.mean(majority_probs_T[i])
            bin_ = 0 if avg<0.5 else 1
            final_avg_output.append(bin_)
            prob.append(avg)

        
        np.savetxt(self.eval_dir + '/' + 'probabilities.csv', majority_probs_T, delimiter=",")
        
        # Calculate metrics for majority voting
        majority_metrics = calculate_metrics(final_labels, final_outputs)
        majority_metrics['Fold'] = 'Majority Voting'
        metrics.append(majority_metrics)    
        
        avg_prob_metrics = calculate_metrics(final_labels, prob)
        avg_prob_metrics['Fold'] = 'Avg Probability'
        metrics.append(avg_prob_metrics) 
            
        name_file = self.eval_dir + '/' + check_ + '_test_metrics.xlsx'
        results = pd.DataFrame(metrics)
        results.to_excel(name_file, index=False)
        
        calibration_plot(final_labels, prob, self.eval_dir)
        
        