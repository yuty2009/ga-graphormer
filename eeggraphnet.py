<<<<<<< HEAD

import torch
import torch.nn as nn
import os, sys; sys.path.append(os.path.dirname(__file__))
from spatial import GNNPooling_pyg, AttnPooling, GraphormerPooling


class EEGGraphNet(nn.Module):
    def __init__(
            self, n_classes, n_channels=32, embed_dim=128, pooling=False, # pooling=True often make the result collapse
            droprate_t=0.5, droprate_s=0.5, droprate_fc=0., spatial_type='gcn', **kwargs
        ):
        super(EEGGraphNet, self).__init__()

        # temporal encoder
        self.encoder_t = PatchMixer(
            n_classes=0, embed_dim=embed_dim, patch_size=kwargs.get('patch_size'), 
            kernel_size=5, droprate_mixer=droprate_t)
        # self.encoder_t = EEGChannel(
        #     n_classes=0, n_timepoints=kwargs.get('n_timepoints'), dropout=droprate_t,
        #     n_filters_1=kwargs.get('n_filters_1', 8),
        #     filter_size_time_1=kwargs.get('filter_size_time_1', 50),
        #     n_filters_2=embed_dim,
        #     filter_size_time_2=kwargs.get('filter_size_time_2', 22),
        # )
        feature_dim = self.encoder_t.feature_dim
        
        self.proj = nn.Linear(feature_dim, embed_dim) if feature_dim != embed_dim else nn.Identity()

        # spatial encoder
        adj = torch.eye(n_channels).numpy() if kwargs.get('adj') is None else kwargs.get('adj')
        if spatial_type == 'gcn':
            self.encoder_s = GNNPooling_pyg(
                n_channels=n_channels, input_dim=embed_dim, embed_dim=embed_dim, n_layers=5, 
                droprate=droprate_s, pooling=pooling, adj=adj)
        elif spatial_type == 'graphormer':
            self.encoder_s = AttnPooling(
                n_channels=n_channels, embed_dim=embed_dim, n_heads=4, droprate=0., pooling=pooling, adj=adj)
            # self.encoder_s = GraphformerPooling(
            #     n_channels=n_channels, embed_dim=embed_dim, n_layers=2, n_heads=4, pooling=pooling, droprate_input=0., adj=adj
            # )
        else:
            raise ValueError(f"Unknown spatial type: {spatial_type}")

        # classfier head
        self.drop_fc = nn.Dropout(droprate_fc)
        self.feature_dim = embed_dim if pooling else n_channels * embed_dim
        self.fc = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()  

    def forward(self, x, mask=None):
        # x: (bs, 1, n_timepoints, n_channels)
        inshape = x.shape
        x = x.permute(0, 3, 1, 2) # (bs, n_channels, 1, n_timepoints)
        x = x.reshape(-1, *x.shape[2:]) # (bs*n_channels, 1, n_timepoints)
        x = self.encoder_t(x) # (bs*n_channels, feature_dim)
        x = self.proj(x) # (bs*n_channels, embed_dim)
        x = x.reshape(inshape[0], inshape[-1], -1) # (bs, n_channels, embed_dim)
        outs = self.encoder_s(x, mask) # (bs, embed_dim) or (bs, n_channels*embed_dim)
        attn_weights = None
        if isinstance(outs, tuple):
            x, attn_weights = outs
        x = x.flatten(1)
        x = self.drop_fc(x)
        x = self.fc(x)
        return x, attn_weights
    

class EEGChannel(nn.Module):
    def __init__(
        self, n_classes, n_timepoints, dropout = 0.5,
        n_filters_1 = 8, filter_size_time_1 = 125, 
        pool_size_time_1 = 4, pool_stride_time_1 = 4,
        n_filters_2 = 16, filter_size_time_2 = 22,
        pool_size_time_2 = 8, pool_stride_time_2 = 8,
    ):
        super().__init__()
        assert filter_size_time_1 <= n_timepoints, "Temporal filter size error"

        self.features = nn.Sequential(
            # temporal filtering
            nn.Conv1d(1, n_filters_1, filter_size_time_1, padding=filter_size_time_1//2, bias=False),
            nn.BatchNorm1d(n_filters_1),
            nn.AvgPool1d(pool_size_time_1, stride=pool_stride_time_1),
            nn.Dropout(dropout),
            # Separable Convolution
            SeparableConv1d(
                n_filters_1, n_filters_2, filter_size_time_2, padding=filter_size_time_2//2, bias=False),
            nn.BatchNorm1d(n_filters_2),
            nn.ELU(),
            nn.AvgPool1d(pool_size_time_2, stride=pool_stride_time_2),
        )

        n_features_1 = (n_timepoints - pool_size_time_1)//pool_stride_time_1 + 1
        n_features_2 = (n_features_1 - pool_size_time_2)//pool_stride_time_2 + 1
        self.feature_dim = n_filters_2 * n_features_2

        self.fc = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: (bs, 1, n_timepoints)
        x = self.features(x) # (bs, n_filters_2, n_features_2)
        x = x.view(x.size(0), -1) # (bs, n_filters_2*n_features_2)
        x = self.fc(x)
        return x
    
class SeparableConv1d(torch.nn.Module):
    """
    https://gist.github.com/bdsaglam/
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = torch.nn.Conv1d(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=in_channels, # key point 1
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = torch.nn.Conv1d(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1, # key point 2
             stride=1,
             padding=0,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )
    
    def forward(self, x):
        return self.pointConv(self.spatialConv(x))
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
    
class PatchMixer(nn.Module):
    def __init__(
            self, n_classes=0, embed_dim=128, patch_size=10, 
            n_layers=3, kernel_size=5, droprate_mixer=0.25, droprate_fc=0.,
        ):
        super().__init__()
        # temporal encoder
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self.feature_encoder = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size, groups=embed_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm1d(embed_dim)
                )),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(droprate_mixer),
            ) for _ in range(n_layers)],
            nn.AdaptiveAvgPool1d((1)),
        )
        self.feature_dim = embed_dim
        # classfier head
        self.dropout = nn.Dropout(droprate_fc)
        self.fc = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()      

    def forward(self, x):          
        # x: (bs, 1, n_timepoints)
        x = self.patch_embed(x) # (bs, embed_dim, n_patches)
        x = self.feature_encoder(x) # (bs, embed_dim, 1)
        x = x.squeeze(-1) # (bs, embed_dim)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':

    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    # from torch_geometric.loader import DataLoader
    from eegchan import DEAP_CHANNEL_LIST, DEAP_ADJACENCY_MATRIX
    from eegdataset import ToTensor, MaskedEEGDataset, MaskedGraphEEGDataset

    n_classes = 2
    n_epochs = 50
    n_channels = 32
    n_timepoints = 256
    adj = np.array(DEAP_ADJACENCY_MATRIX)
    full_channels = DEAP_CHANNEL_LIST
    used_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

    epochs = torch.randn(n_epochs, n_timepoints, len(used_channels))
    targets = torch.randint(0, n_classes, (epochs.size(0),))

    dataset = MaskedEEGDataset(
        epochs, targets, ToTensor(),
        chlabels=used_channels, 
        full_chlabels=full_channels
    )

    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    channel_masks = torch.ones(n_epochs, n_channels, dtype=torch.bool)
    for i, ch in enumerate(full_channels):
        if ch not in used_channels:
            channel_masks[:, i] = False

    model = EEGGraphNet(
        n_classes=2, n_channels=n_channels, embed_dim=128, 
        n_layers=3, n_timepoints=n_timepoints, adj=adj, 
        spatial_type='graphormer', patch_size=16, pooling=False,
    )
    
    for data, target, channel_mask in loader:
        print(data.shape)
        out = model(data)
        if isinstance(out, tuple):
            out, attn_weights = out
        print(out.shape)
        break
=======

import torch
import torch.nn as nn
import os, sys; sys.path.append(os.path.dirname(__file__))
from spatial import GNNPooling_pyg, AttnPooling, GraphormerPooling


class EEGGraphNet(nn.Module):
    def __init__(
            self, n_classes, n_channels=32, embed_dim=128, pooling=False, # pooling=True often make the result collapse
            droprate_t=0.5, droprate_s=0.5, droprate_fc=0., spatial_type='gcn', **kwargs
        ):
        super(EEGGraphNet, self).__init__()

        # temporal encoder
        self.encoder_t = PatchMixer(
            n_classes=0, embed_dim=embed_dim, patch_size=kwargs.get('patch_size'), 
            kernel_size=5, droprate_mixer=droprate_t)
        # self.encoder_t = EEGChannel(
        #     n_classes=0, n_timepoints=kwargs.get('n_timepoints'), dropout=droprate_t,
        #     n_filters_1=kwargs.get('n_filters_1', 8),
        #     filter_size_time_1=kwargs.get('filter_size_time_1', 50),
        #     n_filters_2=embed_dim,
        #     filter_size_time_2=kwargs.get('filter_size_time_2', 22),
        # )
        feature_dim = self.encoder_t.feature_dim
        
        self.proj = nn.Linear(feature_dim, embed_dim) if feature_dim != embed_dim else nn.Identity()

        # spatial encoder
        adj = torch.eye(n_channels).numpy() if kwargs.get('adj') is None else kwargs.get('adj')
        if spatial_type == 'gcn':
            self.encoder_s = GNNPooling_pyg(
                n_channels=n_channels, input_dim=embed_dim, embed_dim=embed_dim, n_layers=5, 
                droprate=droprate_s, pooling=pooling, adj=adj)
        elif spatial_type == 'graphormer':
            self.encoder_s = AttnPooling(
                n_channels=n_channels, embed_dim=embed_dim, n_heads=4, droprate=0., pooling=pooling, adj=adj)
            # self.encoder_s = GraphformerPooling(
            #     n_channels=n_channels, embed_dim=embed_dim, n_layers=2, n_heads=4, pooling=pooling, droprate_input=0., adj=adj
            # )
        else:
            raise ValueError(f"Unknown spatial type: {spatial_type}")

        # classfier head
        self.drop_fc = nn.Dropout(droprate_fc)
        self.feature_dim = embed_dim if pooling else n_channels * embed_dim
        self.fc = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()  

    def forward(self, x, mask=None):
        # x: (bs, 1, n_timepoints, n_channels)
        inshape = x.shape
        x = x.permute(0, 3, 1, 2) # (bs, n_channels, 1, n_timepoints)
        x = x.reshape(-1, *x.shape[2:]) # (bs*n_channels, 1, n_timepoints)
        x = self.encoder_t(x) # (bs*n_channels, feature_dim)
        x = self.proj(x) # (bs*n_channels, embed_dim)
        x = x.reshape(inshape[0], inshape[-1], -1) # (bs, n_channels, embed_dim)
        outs = self.encoder_s(x, mask) # (bs, embed_dim) or (bs, n_channels*embed_dim)
        attn_weights = None
        if isinstance(outs, tuple):
            x, attn_weights = outs
        x = x.flatten(1)
        x = self.drop_fc(x)
        x = self.fc(x)
        return x, attn_weights
    

class EEGChannel(nn.Module):
    def __init__(
        self, n_classes, n_timepoints, dropout = 0.5,
        n_filters_1 = 8, filter_size_time_1 = 125, 
        pool_size_time_1 = 4, pool_stride_time_1 = 4,
        n_filters_2 = 16, filter_size_time_2 = 22,
        pool_size_time_2 = 8, pool_stride_time_2 = 8,
    ):
        super().__init__()
        assert filter_size_time_1 <= n_timepoints, "Temporal filter size error"

        self.features = nn.Sequential(
            # temporal filtering
            nn.Conv1d(1, n_filters_1, filter_size_time_1, padding=filter_size_time_1//2, bias=False),
            nn.BatchNorm1d(n_filters_1),
            nn.AvgPool1d(pool_size_time_1, stride=pool_stride_time_1),
            nn.Dropout(dropout),
            # Separable Convolution
            SeparableConv1d(
                n_filters_1, n_filters_2, filter_size_time_2, padding=filter_size_time_2//2, bias=False),
            nn.BatchNorm1d(n_filters_2),
            nn.ELU(),
            nn.AvgPool1d(pool_size_time_2, stride=pool_stride_time_2),
        )

        n_features_1 = (n_timepoints - pool_size_time_1)//pool_stride_time_1 + 1
        n_features_2 = (n_features_1 - pool_size_time_2)//pool_stride_time_2 + 1
        self.feature_dim = n_filters_2 * n_features_2

        self.fc = nn.Linear(self.feature_dim, n_classes) if n_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: (bs, 1, n_timepoints)
        x = self.features(x) # (bs, n_filters_2, n_features_2)
        x = x.view(x.size(0), -1) # (bs, n_filters_2*n_features_2)
        x = self.fc(x)
        return x
    
class SeparableConv1d(torch.nn.Module):
    """
    https://gist.github.com/bdsaglam/
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = torch.nn.Conv1d(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=in_channels, # key point 1
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = torch.nn.Conv1d(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1, # key point 2
             stride=1,
             padding=0,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )
    
    def forward(self, x):
        return self.pointConv(self.spatialConv(x))
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
    
class PatchMixer(nn.Module):
    def __init__(
            self, n_classes=0, embed_dim=128, patch_size=10, 
            n_layers=3, kernel_size=5, droprate_mixer=0.25, droprate_fc=0.,
        ):
        super().__init__()
        # temporal encoder
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self.feature_encoder = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size, groups=embed_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm1d(embed_dim)
                )),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(droprate_mixer),
            ) for _ in range(n_layers)],
            nn.AdaptiveAvgPool1d((1)),
        )
        self.feature_dim = embed_dim
        # classfier head
        self.dropout = nn.Dropout(droprate_fc)
        self.fc = nn.Linear(embed_dim, n_classes) if n_classes > 0 else nn.Identity()      

    def forward(self, x):          
        # x: (bs, 1, n_timepoints)
        x = self.patch_embed(x) # (bs, embed_dim, n_patches)
        x = self.feature_encoder(x) # (bs, embed_dim, 1)
        x = x.squeeze(-1) # (bs, embed_dim)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':

    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    # from torch_geometric.loader import DataLoader
    from eegchan import DEAP_CHANNEL_LIST, DEAP_ADJACENCY_MATRIX
    from eegdataset import ToTensor, MaskedEEGDataset, MaskedGraphEEGDataset

    n_classes = 2
    n_epochs = 50
    n_channels = 32
    n_timepoints = 256
    adj = np.array(DEAP_ADJACENCY_MATRIX)
    full_channels = DEAP_CHANNEL_LIST
    used_channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']

    epochs = torch.randn(n_epochs, n_timepoints, len(used_channels))
    targets = torch.randint(0, n_classes, (epochs.size(0),))

    dataset = MaskedEEGDataset(
        epochs, targets, ToTensor(),
        chlabels=used_channels, 
        full_chlabels=full_channels
    )

    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    channel_masks = torch.ones(n_epochs, n_channels, dtype=torch.bool)
    for i, ch in enumerate(full_channels):
        if ch not in used_channels:
            channel_masks[:, i] = False

    model = EEGGraphNet(
        n_classes=2, n_channels=n_channels, embed_dim=128, 
        n_layers=3, n_timepoints=n_timepoints, adj=adj, 
        spatial_type='graphormer', patch_size=16, pooling=False,
    )
    
    for data, target, channel_mask in loader:
        print(data.shape)
        out = model(data)
        if isinstance(out, tuple):
            out, attn_weights = out
        print(out.shape)
        break
>>>>>>> de32b1275879a7d77f6de917d6521e87e2f591e4
