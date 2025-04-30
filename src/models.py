import torch
from torch import nn
from monai.networks.layers.factories import Act
from monai.networks.blocks import UpSample, MaxAvgPool
from src.ops import *
from monai.networks.nets import DenseNet121
from src.resnet_blocks.util import ResNetFeatures

#https://github.com/titu1994/LSTM-FCN/blob/master/utils/layer_utils.py 
#https://www.sciencedirect.com/science/article/pii/S1077314223001194

class ConvLSTM_class(nn.Module): 

    '''
    here we input the 3D images directly to the convolutional LSTM, after that we have two 3D conv and fully connected layers for classification
    '''

    def __init__(self, spatial_dims, in_channels, out_channels, len_sequence, kernel_size, device):
    
        super(ConvLSTM_class, self).__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.len_sequence = len_sequence 
        self.kernel_size = kernel_size
        self.device = device
        self.track_changes = ConvLSTM(input_dim=self.in_channels, #one channel as input
                                               hidden_dim=16,
                                               kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                               num_layers=self.len_sequence,
                                               batch_first=True,
                                               bias=True).to(device)

        self.classifier = Depth_convolution(n_in=16, kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size), n_classes=out_channels).to(device)

  
    def forward(self, x):
        x = x.permute(0, 1, 2, 5, 3, 4) #the depth is before H,W for torch 3Conv
        #(batch, time, channel, d,h,w)
        _, last_states = self.track_changes(input_tensor = x)
        outp = self.classifier(input_tensor = last_states[0][0])
        
        return outp 


class ConvLSTM_class2(nn.Module): 

    '''
    here we input the 3D images directly to the convolutional LSTM, after that we have two 3D conv and fully connected layers for classification
    '''

    def __init__(self, spatial_dims, in_channels, out_channels, len_sequence, kernel_size, device):
    
        super(ConvLSTM_class2, self).__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.len_sequence = len_sequence 
        self.kernel_size = kernel_size
        self.device = device
        self.track_changes = ConvLSTM(input_dim=self.in_channels, #one channel as input
                                               hidden_dim=16,
                                               kernel_size=(self.kernel_size//2, self.kernel_size, self.kernel_size),
                                               num_layers=self.len_sequence,
                                               batch_first=True,
                                               bias=True).to(device)

        self.classifier = Depth_convolution2(n_in=16, kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size), n_classes=out_channels).to(device)

  
    def forward(self, x):
        x = x.permute(0, 1, 2, 5, 3, 4) #the depth is before H,W for torch 3Conv
        #(batch, time, channel, d,h,w)
        _, last_states = self.track_changes(input_tensor = x)
        outp = self.classifier(input_tensor = last_states[0][0])
        
        return outp 


class MSResNet(nn.Module): 

    '''
    here we first extract features from the 3D images separately, then we model changes using the convolutional LSTM, 
    after that we fully connected layers for classification
    
    '''

    def __init__(self, batch_size, spatial_dims, in_channels, out_channels, len_sequence, kernel_size, device):
    
        super(MSResNet, self).__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.len_sequence = len_sequence 
        self.kernel_size = kernel_size
        self.device = device
        last_dim = self.len_sequence * 24 * 24 * 24 

        self.singleT_features = ResNet_multiple(input_dim=self.in_channels, #one channel as input
                                            kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                            len_sequence = self.len_sequence).to(device)
        #shape (1,1,3,12,24,24)
        self.longitudinal_features = LongitudinalSelfAttention().to(device)
        
        self.classifier = Classifier(input_dim = last_dim, num_classes=out_channels, dropout=True).to(device)

  
    def forward(self, x):
    
        x = x.permute(0, 1, 2, 5, 3, 4) #the depth is before H,W for torch 3Conv
        #(batch, time, channel, d,h,w)
        x = self.singleT_features(x) #each time point is processed separately
        x = self.longitudinal_features(x)
        outp = self.classifier(x)
        
        return outp 

class MSResNet_smaller(nn.Module): 

    '''
    here we first extract features from the 3D images separately, then we model changes using the convolutional LSTM, 
    after that we fully connected layers for classification
    
    '''

    def __init__(self, batch_size, spatial_dims, in_channels, out_channels, len_sequence, kernel_size, device):
    
        super(MSResNet_smaller, self).__init__()
        
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.len_sequence = len_sequence 
        self.kernel_size = kernel_size
        self.device = device
        last_dim = self.len_sequence * 24 * 24 * 24 

        self.singleT_features = ResNet_multiple_smaller(input_dim=self.in_channels, #one channel as input
                                            kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
                                            len_sequence = self.len_sequence).to(device)

        self.longitudinal_features = LongitudinalSelfAttention().to(device)
        
        self.classifier = Classifier(input_dim = last_dim, num_classes=out_channels, dropout=True).to(device)

  
    def forward(self, x):
    
        x = x.permute(0, 1, 2, 5, 3, 4) #the depth is before H,W for torch 3Conv
        #(batch, time, channel, d,h,w)
        x = self.singleT_features(x) #each time point is processed separately
        x = self.longitudinal_features(x)
        outp = self.classifier(x)
        
        return outp 
    
    
class DensNet_selfAtt(nn.Module): 

    def __init__(self, num_classes, in_channels=1):
    
        super(DensNet_selfAtt, self).__init__()
        
        # Define a single 3D DenseNet for feature extraction
        self.resnet = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2)

        # Define Self-Attention for temporal processing
        self.self_attention = LongitudinalSelfAttention()

        # Fully connected layer for classification
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        bs = x.size(0)  # Batch size
        features = []

        # Iterate over time stamps
        for i in range(x.size(2)):  # x.size(2) is the number of time steps
            # Extract features for each time stamp using the shared 3D DenseNet
            features_ = self.resnet.features(x[:, :, i])  # Use the shared DenseNet model
            # print(features_.shape)
            
            features_ = self.avgpool(features_)  # Apply average pooling
            # print(features_.shape)
            features.append(features_.view(bs, -1))  # Flatten and collect features

        features = torch.stack(features, dim=1)  # Stack features along the time dimension

        # Perform temporal processing with Self-Attention
        features = self.self_attention(features)

        # Aggregate features by mean pooling over time dimension
        features = features.mean(dim=1)

        # Fully connected layer for classification
        output = self.fc1(features)
        output = self.relu1(output)
        output = self.fc2(output)
        
        return F.log_softmax(output, dim=1)


class ResNet_selfAtt(nn.Module): 

    def __init__(self, num_classes, in_channels=1):
    
        super(ResNet_selfAtt, self).__init__()
        
        # Define a single 3D DenseNet for feature extraction
        self.resnet = ResNetFeatures('resnet18',spatial_dims=3, in_channels=in_channels,pretrained=False)

        # Define Self-Attention for temporal processing
        self.self_attention = LongitudinalSelfAttention()

        # Fully connected layer for classification
        self.fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        bs = x.size(0)  # Batch size
        features = []

        # Iterate over time stamps
        for i in range(x.size(2)):  # x.size(2) is the number of time steps
            # Extract features for each time stamp using the shared 3D DenseNet
            features_ = self.resnet(x[:, :, i])  # Use the shared DenseNet model
            # print(len(features_),features_[-1].shape)
            features_ = self.avgpool(features_[-1])  # Apply average pooling
            # print(features_.shape)
            features.append(features_[:,:].view(bs, -1))  # Flatten and collect features

        features = torch.stack(features, dim=1)  # Stack features along the time dimension

        # Perform temporal processing with Self-Attention
        features = self.self_attention(features)

        # Aggregate features by mean pooling over time dimension
        features = features.mean(dim=1)

        # Fully connected layer for classification
        output = self.fc1(features)
        output = self.relu1(output)
        output = self.fc2(output)
        
        return F.log_softmax(output, dim=1)
    

class SEResNet_selfAtt(nn.Module): 

    def __init__(self, num_classes, in_channels=1):
    
        super(SEResNet_selfAtt, self).__init__()
        
        self.resnet = monai.networks.nets.SENet(
                            spatial_dims=3,
                            in_channels=in_channels,  # or 3 if your data has 3 channels
                            block=monai.networks.blocks.SEResNetBottleneck,
                            layers=(3, 4, 6, 3), groups=1, reduction=16, 
                            dropout_prob=None, inplanes=64, 
                            downsample_kernel_size=1, input_3x3=False # Change this according to your number of classes
                        )

        # Define Self-Attention for temporal processing
        # self.self_attention = SelfAttention(2048)
        self.self_attention = LongitudinalSelfAttention()

        # Fully connected layer for classification
        self.fc1 = nn.Linear(2048, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        bs = x.size(0)  # Batch size
        features = []

        # Iterate over time stamps
        for i in range(x.size(2)):  # x.size(2) is the number of time steps
            # Extract features for each time stamp using the shared 3D DenseNet
            features_ = self.resnet.features(x[:, :, i])  # Use the shared DenseNet model
            
            features_ = self.avgpool(features_)  # Apply average pooling
            features.append(features_.view(bs, -1))  # Flatten and collect features
            # print(features_.shape)

        features = torch.stack(features, dim=1)  # Stack features along the time dimension

        # Perform temporal processing with Self-Attention
        features = self.self_attention(features)

        # Aggregate features by mean pooling over time dimension
        features = features.mean(dim=1)

        # Fully connected layer for classification
        output = self.fc1(features)
        output = self.relu1(output)
        output = self.fc2(output)
        
        return F.log_softmax(output, dim=1)


class DensNet_TCN(nn.Module): 

    def __init__(self, num_classes, in_channels=1):
    
        super(DensNet_TCN, self).__init__()
        
        # Define a single 3D DenseNet for feature extraction
        self.resnet = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2)
        
        self.tcn = TCN(input_size=1024, output_size=num_classes, num_channels=[256, 128, 64], kernel_size=3, dropout=0.2)
        
        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        bs = x.size(0)  # Batch size
        features = []

        # Iterate over time stamps
        for i in range(x.size(2)):  # x.size(2) is the number of time steps
            # Extract features for each time stamp using the shared 3D DenseNet
            features_ = self.resnet.features(x[:, :, i])  # Use the shared DenseNet model
            # print(features_.shape)
            
            features_ = self.avgpool(features_)  # Apply average pooling
            features.append(features_)  # Flatten and collect features
        
        features = torch.stack(features, dim=2)  # Stack features along the time dimension
        features = features.view(bs, 1024,-1)
        # Perform temporal processing with Self-Attention
        output = self.tcn(features)
        return output

class ResNet_TCN(nn.Module): 

    def __init__(self, num_classes, in_channels=1):
    
        super(ResNet_TCN, self).__init__()
        
        # Define a single 3D DenseNet for feature extraction
        self.resnet = ResNetFeatures('resnet18',spatial_dims=3, in_channels=in_channels,pretrained=False)

        self.tcn = TCN(input_size=512, output_size=num_classes, num_channels=[256, 128, 64], kernel_size=3, dropout=0.2)
        
        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        bs = x.size(0)  # Batch size
        features = []

        # Iterate over time stamps
        for i in range(x.size(2)):  # x.size(2) is the number of time steps
            # Extract features for each time stamp using the shared 3D DenseNet
            features_ = self.resnet(x[:, :, i])  # Use the shared DenseNet model
            # print(features_.shape)
            features_ = self.avgpool(features_[-1])  # Apply average pooling
            features.append(features_)  # Flatten and collect features
        
        features = torch.stack(features, dim=2)  # Stack features along the time dimension
        
        features = features.view(bs, 512,-1)
        # Perform temporal processing with Self-Attention
        output = self.tcn(features)
        return output
    

class SEResNet_TCN(nn.Module): 

    def __init__(self, num_classes, in_channels=1):
    
        super(SEResNet_TCN, self).__init__()
        
        self.resnet = monai.networks.nets.SENet(
                            spatial_dims=3,
                            in_channels=in_channels,  # or 3 if your data has 3 channels
                            block=monai.networks.blocks.SEResNetBottleneck,
                            layers=(3, 4, 6, 3), groups=1, reduction=16, 
                            dropout_prob=None, inplanes=64, 
                            downsample_kernel_size=1, input_3x3=False # Change this according to your number of classes
                        )

        self.tcn = TCN(input_size=1024, output_size=num_classes, num_channels=[256, 128, 64], kernel_size=3, dropout=0.2)
        
        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        bs = x.size(0)  # Batch size
        features = []

        # Iterate over time stamps
        for i in range(x.size(2)):  # x.size(2) is the number of time steps
            # Extract features for each time stamp using the shared 3D DenseNet
            features_ = self.resnet.features(x[:, :, i])  # Use the shared DenseNet model
            # print(features_.shape)
            
            features_ = self.avgpool(features_)  # Apply average pooling
            features.append(features_)  # Flatten and collect features
        
        features = torch.stack(features, dim=2)  # Stack features along the time dimension
        features = features.view(bs, 1024,-1)
        # Perform temporal processing with Self-Attention
        output = self.tcn(features)
        return output
