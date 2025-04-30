import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np
import torch
from torch import nn
import random
import os
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.calibration import calibration_curve

def print_image(batch_data, path):
    slice_index = 40  # Adjust this as needed

    # Select one image from the batch (index 0)
    input_tensor = batch_data[0]  # Assuming you want to select the first image in the batch
    input_tensor = input_tensor.squeeze()
    
    # Create a subplot grid based on the number of time points
    num_time_points = input_tensor.shape[0]
    num_cols = 1  # Number of columns in the grid
    num_rows = num_time_points  # Calculate number of rows

    # Set up the plot
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))  # Adjust figsize as needed

    # If there's only one row, axs is not a list of lists, but just a list
    if num_time_points == 1:
        axs = [axs]

    # Iterate over the time points
    for t in range(num_time_points):
        input_slice = input_tensor[t, :, :, slice_index]  # Assuming depth dimension is 3
        slice = input_slice.detach().cpu().numpy()
        
        row_idx = t  # Since we have only one column, row index is just t
        col_idx = 0  # Only one column

        axs[row_idx].imshow(slice, cmap='gray')  # Assuming grayscale images
        axs[row_idx].axis('off')

    plt.tight_layout()
    plt.savefig(path+"/test.png")
    plt.close()
    
    
def calculate_metrics(labels, probs, average='macro'):
    
    outputs = [0 if i < 0.5 else 1 for i in probs]
    
    f1 = f1_score(labels, outputs, average=average)
    accuracy = accuracy_score(labels, outputs)
    auc = roc_auc_score(labels, probs, multi_class='ovr')  # Adjust if multiclass
    precision = precision_score(labels, outputs, average=average)
    recall = recall_score(labels, outputs, average=average)
    
    return {
        'F1 Score Macro': f1,
        'Accuracy': accuracy,
        'AUC OVR': auc,
        'Precision Macro': precision,
        'Recall Macro': recall,
    }
    
    
    
def calibration_plot(labels, outcome, path):
    
    plt.figure(figsize=(10, 7))
    prob_true, prob_pred = calibration_curve(labels, outcome, n_bins=10)

    plt.plot(prob_pred, prob_true, marker='o', label='Deep Learning Model', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('2-Year DFS Calibration Plot')
    plt.legend()
    name_file = path + '/calibration_plot.jpeg'
    plt.savefig(name_file)
    plt.close()
    

def calculate_sample_weights(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    return sample_weights


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        #checkpoint = {"config": Config, "model_state_dict": model.state_dict()}
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        if self.path != None:
            torch.save(model.state_dict(), self.path)
            
        self.val_loss_min = val_loss
        
def seed_everything(seed=42):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def initialize_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight.data, nonlinearity="relu")
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.Conv3d): 
        nn.init.kaiming_uniform_(model.weight.data, nonlinearity="relu")
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)
            
def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
        
    