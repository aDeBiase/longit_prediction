import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import monai
from functools import partial

#https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
#did not use following classes yet
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        out = x[:, :, :-self.chomp_size].contiguous()
        return out


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


#####
#https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
# I am changing this
class ConvLSTMCell(nn.Module):
     # N,T,C,H,W,D = 1, time, channels, H, W, D

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim #we have one channel now
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = 'same' #kernel_size[0]// 2, kernel_size[1]// 2, kernel_size[2] // 2 #// 2
        self.bias = bias
        
        self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels= 4 * self.hidden_dim, #
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        depth, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, D, H, W or T, B, C, D, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            #first time the input dim will be 1, next times will be the num of hidd layers
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w, d) or (b, t, c, h, w, d)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, d, h, w) -> (b, t, c, d, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4, 5)
        
        b, _, _, d, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(d, h, w))
        
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        
        cur_layer_input = input_tensor
    
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            #for each time point in the sequence
            for t in range(seq_len):
                
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :, :],
                                                 cur_state=[h, c])

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


## 
class Depth_convolution(nn.Module):

    def __init__(self,n_in,kernel_size,n_classes):

        super(Depth_convolution,self).__init__()
        
        self.n_in = n_in #time steps
        self.n_classes = n_classes
        self.kernel_size = kernel_size

        self.depth_conv1 = nn.Conv3d(self.n_in,16,kernel_size=self.kernel_size,stride=1,groups=self.n_in)
        self.depth_conv2 = nn.Conv3d(16,64,kernel_size=self.kernel_size,stride=1,groups=16)

        #self.fc1 = nn.Linear(in_features=16, out_features=64)
        
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=self.n_classes)


    def forward(self, input_tensor: torch.Tensor):

        x = input_tensor
        
        x = F.relu(self.depth_conv1(x))
        x = F.relu(self.depth_conv2(x))

        x = F.avg_pool3d(x,kernel_size=x.shape[2:])

        x = x.squeeze(dim=2).squeeze(dim=2).squeeze(dim=2)

        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.log_softmax(x,1)


class Depth_convolution2(nn.Module):
    
    '''
    The depthwise convolutions are implemented in pytorch in the Conv modules with the group parameter.
    For an input of c channels, and depth multiplier of d, the nn.Conv2d parameters become
    in_channels = c
    out_channels = d*c
    groups = c
    '''

    def __init__(self,n_in,kernel_size,n_classes):

        super(Depth_convolution2,self).__init__()
        
        self.n_in = n_in #time steps
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        
        #two depthwise convolutions
        self.depth_conv1 = nn.Conv3d(self.n_in,2*self.n_in,kernel_size=self.kernel_size,stride=1,groups=self.n_in)
        self.depth_conv2 = nn.Conv3d(2*self.n_in,4*self.n_in,kernel_size=self.kernel_size,stride=1,groups=2*self.n_in)

        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=self.n_classes)


    def forward(self, input_tensor: torch.Tensor):

        x = input_tensor
        
        x = F.relu(self.depth_conv1(x))
        x = F.relu(self.depth_conv2(x))

        x = F.avg_pool3d(x,kernel_size=x.shape[2:])

        x = x.squeeze(dim=2).squeeze(dim=2).squeeze(dim=2)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.log_softmax(x,1)


class ResidualBlock(nn.Module):
    def __init__(self, n_in, kernel_size):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(num_features=n_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=n_in, out_channels=n_in, kernel_size=kernel_size, stride=1,padding="same")

        self.bn2 = nn.BatchNorm3d(num_features=n_in)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=n_in, out_channels=n_in, kernel_size=kernel_size, stride=1,padding="same")

        self.bn3 = nn.BatchNorm3d(num_features=n_in)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels=n_in, out_channels=n_in, kernel_size=kernel_size, stride=1,padding="same")

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)  # Adding max pooling layer

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual  # Adding the residual connection
        out = self.relu3(out)

        out = self.maxpool(out)  # Applying max pooling
        return out    


class ResidualBlock_smaller(nn.Module):
    def __init__(self, n_in, kernel_size):
        super(ResidualBlock_smaller, self).__init__()

        self.bn1 = nn.BatchNorm3d(num_features=n_in)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=n_in, out_channels=n_in, kernel_size=kernel_size, stride=1,padding="same")

        self.bn3 = nn.BatchNorm3d(num_features=n_in)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(in_channels=n_in, out_channels=n_in, kernel_size=kernel_size, stride=1,padding="same")

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)  # Adding max pooling layer

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual  # Adding the residual connection
        out = self.relu3(out)

        out = self.maxpool(out)  # Applying max pooling
        return out  


class ResNet_multiple(nn.Module):
    
    def __init__(self,input_dim,kernel_size,len_sequence):

        super(ResNet_multiple,self).__init__()
        
        self.n_in = input_dim #time steps
        self.kernel_size = kernel_size
        self.len_sequence = len_sequence
        
        all_ResBlocks = list()
        
        for k in range(self.len_sequence):
            
            self.resnet1 = ResidualBlock(self.n_in, self.kernel_size)
            self.resnet2 = ResidualBlock(self.n_in, self.kernel_size)
            self.resnet3 = ResidualBlock(self.n_in, self.kernel_size)
            
            all_ResBlocks.append([self.resnet1, self.resnet2, self.resnet3]) 
        
        self.cell_list = nn.ModuleList([nn.ModuleList(i) for i in all_ResBlocks])
        
        
    def forward(self, input_tensor: torch.Tensor):

        b, _, _, d, h, w = input_tensor.size()
        
        seq_len = input_tensor.size(1)
        output = list()
        
        for t in range(seq_len):
            
            x = input_tensor[:, t, :, :, :, :]
            
            for layer_idx in range(3):
                x = self.cell_list[t][layer_idx](x)

            output.append(x)
            
        layer_output = torch.stack(output, dim=1)
        
        return layer_output
        

class ResNet_multiple_smaller(nn.Module):
    
    def __init__(self,input_dim,kernel_size,len_sequence):

        super(ResNet_multiple_smaller,self).__init__()
        
        self.n_in = input_dim #time steps
        self.kernel_size = kernel_size
        self.len_sequence = len_sequence
        
        all_ResBlocks = list()
        
        for k in range(self.len_sequence):
            
            self.resnet1 = ResidualBlock_smaller(self.n_in, self.kernel_size)
            self.resnet2 = ResidualBlock_smaller(self.n_in, self.kernel_size)
            self.resnet3 = ResidualBlock_smaller(self.n_in, self.kernel_size)
                        
            all_ResBlocks.append([self.resnet1, self.resnet2, self.resnet3]) 
        
        self.cell_list = nn.ModuleList([nn.ModuleList(i) for i in all_ResBlocks])
        
        
    def forward(self, input_tensor: torch.Tensor):

        b, _, _, d, h, w = input_tensor.size()
        
        seq_len = input_tensor.size(1)
        output = list()
        
        for t in range(seq_len):
            
            x = input_tensor[:, t, :, :, :, :]
            
            for layer_idx in range(3):
                x = self.cell_list[t][layer_idx](x)

            output.append(x)
            
        layer_output = torch.stack(output, dim=1)
        
        return layer_output
    
class ResNet_extract(nn.Module):
    
    def __init__(self,input_dim,kernel_size,len_sequence):

        super(ResNet_multiple,self).__init__()
        
        self.n_in = input_dim #time steps
        self.kernel_size = kernel_size
        self.len_sequence = len_sequence
        
        #self.resnet = 
        
        '''
        for k in range(self.len_sequence):
            
            self.resnet1 = ResidualBlock(self.n_in, self.kernel_size)
            self.resnet2 = ResidualBlock(self.n_in, self.kernel_size)
            self.resnet3 = ResidualBlock(self.n_in, self.kernel_size)
            
            all_ResBlocks.append([self.resnet1, self.resnet2, self.resnet3]) 
        
        self.cell_list = nn.ModuleList([nn.ModuleList(i) for i in all_ResBlocks])
        '''
        
        
    def forward(self, input_tensor: torch.Tensor):

        b, _, _, d, h, w = input_tensor.size()
        
        seq_len = input_tensor.size(1)
        output = list()
        
        for t in range(seq_len):
            
            x = input_tensor[:, t, :, :, :, :]
            
            for layer_idx in range(3):
                x = self.cell_list[t][layer_idx](x)

            output.append(x)
            
        layer_output = torch.stack(output, dim=1)
        
        return layer_output
        

    
class Feature_extractor(nn.Module):
    
    def __init__(self,input_dim,len_sequence):

        super(Feature_extractor,self).__init__()
        
        self.n_in = input_dim #time steps
        self.len_sequence = len_sequence
        
        self.extractor = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, input_tensor: torch.Tensor):

        b, _, _, d, h, w = input_tensor.size()
        
        seq_len = input_tensor.size(1)
        output = list()
        
        for t in range(seq_len):
            
            x = input_tensor[:, t, :, :, :, :] #each time point goes into a Densenet feature extraction
            x = self.extractor.features(x) 
            x = self.avgpool(x)
            output.append(x.view(b, -1))
            
        layer_output = torch.stack(output, dim=1)
        
        return layer_output
        
        
class LongitudinalSelfAttention(nn.Module):
    def __init__(self):
        super(LongitudinalSelfAttention, self).__init__()
        # Softmax layer to compute attention weights
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        
        #shape (1,3,1,12,24,24)
        original = x
        # Reshape to batch_size, time_step, num_channels*height*width*depth
        b = x.size(0)
        t = x.size(1)
        
        K = x.view(b, t, -1)
        Q = x.view(b, t, -1)
        V = x.view(b, t, -1)
        
        # Compute attention scores
        attention = torch.bmm(K.permute(0, 2, 1), Q)
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(V, attention)
        
        # Reshape back to original shape
        out = out.view(original.size())
        
        #shape (1,3,1,12,24,24)
        return original + out
    
    
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.dropout_ = dropout
        # Define classification layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Flatten the input tensor
        
        batch_size = x.size(0)
        num_features = x.numel() // batch_size
        x = x.view(batch_size, num_features)
        x = F.relu(self.fc1(x))
        if self.dropout_:
            x = self.dropout(x) 
        x = self.fc2(x)
        
        return F.log_softmax(x,1)


#https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
