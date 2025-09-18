<<<<<<< HEAD

import torch
import numpy as np
import pandas as pd
from typing import List, Union
from torch_geometric.data import Data


class ToTensor(object):
    """ 
    Turn a (timepoints x channels) or (T, C) epoch into 
    a (depth x timepoints x channels) or (D, T, C) image for torch.nn.Convnd
    """
    def __init__(self, expand_dim=True) -> None:
        self.expand_dim = expand_dim

    def __call__(self, epoch, target=None):
        if isinstance(epoch, np.ndarray):
            epoch = torch.FloatTensor(epoch.copy())
        if self.expand_dim:
            epoch = epoch.unsqueeze(-3)
        if target is not None:
            return epoch, torch.LongTensor(target)
        return epoch
    

class ToG():
    r'''
    # Copy from https://github.com/torcheeg/torcheeg/torcheeg/transforms/pyg/to.py
    .. code-block:: python

        transform = ToG(adj=DEAP_ADJACENCY_MATRIX)
        transform(eeg=np.random.randn(32, 128))['eeg']
        >>> torch_geometric.data.Data

    Args:
        adj (list): An adjacency matrix represented by a 2D array, each element
            in the adjacency matrix represents the electrode-to-electrode edge weight. 
            Please keep the order of electrodes in the rows and columns of the 
            adjacency matrix consistent with the EEG signal to be transformed.
        add_self_loop (bool): Whether to add self-loop edges to the graph. 
            (default: :obj:`True`)
        threshold (float, optional): Used to cut edges when not None. Edges whose
            weights exceed a threshold are retained. (default: :obj:`None`)
        top_k (int, optional): Used to cut edges when not None. Keep the k edges 
            connected to each node with the largest weights. (default: :obj:`None`)
        binary (bool): Whether to binarize the weights on the edges to 0 and 1. 
            If set to True, binarization are done after topk and threshold, 
            the edge weights that still have values are set to 1, 
            otherwise they are set to 0. (default: :obj:`False`)
        complete_graph (bool): Whether to build as a complete graph. If False, 
            only construct edges between electrodes based on non-zero elements; 
            if True, construct variables between all electrodes and set the 
            weight of non-existing edges to 0. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 adj: List[List],
                 add_self_loop: bool = True,
                 threshold: Union[float, None] = None,
                 top_k: Union[int, None] = None,
                 binary: bool = False,
                 complete_graph: bool = False):
        super(ToG, self).__init__()

        self.add_self_loop = add_self_loop
        self.threshold = threshold
        self.top_k = top_k
        self.binary = binary
        self.complete_graph = complete_graph

        adj = torch.tensor(adj).float()

        if add_self_loop:
            adj = adj + torch.eye(adj.shape[0])

        if not self.threshold is None:
            adj[adj < self.threshold] = 0

        if not self.top_k is None:
            rows = []
            for row in adj:
                vals, index = row.topk(self.top_k)
                topk = torch.zeros_like(row)
                topk[index] = vals
                rows.append(topk)
            adj = torch.stack(rows)

        if self.binary:
            adj[adj != 0] = 1.0

        if self.complete_graph:
            adj[adj == 0] = 1e-6

        self.adj = adj.to_sparse()

    def __call__(self,
                 eeg: Union[np.ndarray, torch.Tensor],
                 **kwargs) -> Data:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of 
                [number of electrodes, number of data points].
            
        Returns:
            torch_geometric.data.Data: The graph representation data types 
                that torch_geometric can accept. Nodes correspond to electrodes, 
                and edges are determined via the given adjacency matrix.
        '''
        data = Data(edge_index=self.adj._indices())
        if isinstance(eeg, np.ndarray):
            data.x = torch.from_numpy(eeg).float()
        else:
            data.x = eeg
        data.edge_attr = self.adj._values()

        return data


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, epochs, targets, transforms=None):
        super(EEGDataset, self).__init__()
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, idx):
        return self.epochs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
    

class MaskedEEGDataset(torch.utils.data.Dataset):
    # Channel-Masked EEG Dataset
    # epochs: n_epochs of (n_samples, n_channels_i)
    # targets: n_epochs of (1)
    # chlabels: list of channel labels (n_channels_i)
    # full_chlabels: list of full channel labels (n_channels)
    # channel_mask: list of channel masks (n_channels)
    def __init__(
            self, epochs, targets, transforms=None, chlabels=None, full_chlabels=None, 
        ):
        super(MaskedEEGDataset, self).__init__()
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.targets = torch.LongTensor(targets)

        self.chlabels = chlabels
        self.full_chlabels = full_chlabels
        self.channel_mask = torch.zeros(len(self.full_chlabels), dtype=torch.bool)
        for i, ch in enumerate(self.chlabels):
            if ch in self.full_chlabels:
                ch_idx = full_chlabels.index(ch)
                self.channel_mask[ch_idx] = True
            else:
                print(f"Warning: channel {ch} not found in full channel list")

        used_chlabels = [self.full_chlabels[i] for i in range(len(self.channel_mask)) if self.channel_mask[i]]
        print(f"Using {len(used_chlabels)}/{len(self.full_chlabels)} channels: {used_chlabels}")

    def __getitem__(self, idx):
        epoch = self.epochs[idx]
        n_channels = len(self.full_chlabels)
        # allow for different number of channels in each epoch
        epoch_full = torch.zeros(*epoch.shape[:-1], n_channels)
        ch_idx = 0
        for i, ch in enumerate(self.channel_mask):
            if ch:
                epoch_full[..., i] = epoch[..., ch_idx]
                ch_idx += 1
        return epoch_full, self.targets[idx], self.channel_mask

    def __len__(self):
        return len(self.targets)
    

class MaskedGraphEEGDataset(torch.utils.data.Dataset):
    # Channel-Masked Graph EEG Dataset
    # epochs: list of EEG epochs n_epochs of (n_samples, n_channels_i)
    # targets: list of target labels (n_epochs)
    # chlabels: list of channel labels (n_channels_i)
    # full_chlabels: list of full channel labels (n_channels)
    # full_adj: full adjacency matrix (n_channels x n_channels)
    def __init__(
            self, epochs, targets, transforms=None, chlabels=None, full_chlabels=None, full_adj=None, 
        ):
        super(MaskedGraphEEGDataset, self).__init__()
        tf_ToG = ToG(adj=full_adj)
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.epochs = [tf_ToG(eeg=epoch) for epoch in self.epochs]
        self.targets = torch.LongTensor(targets)

        self.chlabels = chlabels
        self.full_chlabels = full_chlabels
        self.channel_mask = torch.zeros(len(self.full_chlabels), dtype=torch.bool)
        for i, ch in enumerate(self.chlabels):
            if ch in self.full_chlabels:
                ch_idx = full_chlabels.index(ch)
                self.channel_mask[ch_idx] = True
            else:
                print(f"Warning: channel {ch} not found in full channel list")

        used_chlabels = [self.full_chlabels[i] for i in range(len(self.channel_mask)) if self.channel_mask[i]]
        print(f"Using {len(used_chlabels)}/{len(self.full_chlabels)} channels: {used_chlabels}")

    def __getitem__(self, idx):
        data = self.epochs[idx]
        epoch = data.x
        n_channels = len(self.full_chlabels)
        # allow for different number of channels in each epoch
        epoch_full = torch.zeros(*epoch.shape[:-1], n_channels)
        ch_idx = 0
        for i, ch in enumerate(self.channel_mask):
            if ch:
                epoch_full[..., i] = epoch[..., ch_idx]
                ch_idx += 1
        data.x = epoch_full.transpose(-1, -2) # (n_samples, n_channels, n_timepoints)
        data.channel_mask = self.channel_mask
        data.y = self.targets[idx]
        return data

    def __len__(self):
        return len(self.targets)
    

if __name__ == '__main__':
    
    from torch_geometric.loader import DataLoader

    x = torch.rand(10, 256, 6)
    y = torch.randint(0, 4, (10,))
    dataset = MaskedGraphEEGDataset(
        x, y, 
        chlabels=['F3', 'F4', 'C3', 'C4', 'O1', 'O2'], 
        full_chlabels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'], 
        full_adj=np.eye(10),
    )
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for data in loader:
        print(data)
        break
=======

import torch
import numpy as np
import pandas as pd
from typing import List, Union
from torch_geometric.data import Data


class ToTensor(object):
    """ 
    Turn a (timepoints x channels) or (T, C) epoch into 
    a (depth x timepoints x channels) or (D, T, C) image for torch.nn.Convnd
    """
    def __init__(self, expand_dim=True) -> None:
        self.expand_dim = expand_dim

    def __call__(self, epoch, target=None):
        if isinstance(epoch, np.ndarray):
            epoch = torch.FloatTensor(epoch.copy())
        if self.expand_dim:
            epoch = epoch.unsqueeze(-3)
        if target is not None:
            return epoch, torch.LongTensor(target)
        return epoch
    

class ToG():
    r'''
    # Copy from https://github.com/torcheeg/torcheeg/torcheeg/transforms/pyg/to.py
    .. code-block:: python

        transform = ToG(adj=DEAP_ADJACENCY_MATRIX)
        transform(eeg=np.random.randn(32, 128))['eeg']
        >>> torch_geometric.data.Data

    Args:
        adj (list): An adjacency matrix represented by a 2D array, each element
            in the adjacency matrix represents the electrode-to-electrode edge weight. 
            Please keep the order of electrodes in the rows and columns of the 
            adjacency matrix consistent with the EEG signal to be transformed.
        add_self_loop (bool): Whether to add self-loop edges to the graph. 
            (default: :obj:`True`)
        threshold (float, optional): Used to cut edges when not None. Edges whose
            weights exceed a threshold are retained. (default: :obj:`None`)
        top_k (int, optional): Used to cut edges when not None. Keep the k edges 
            connected to each node with the largest weights. (default: :obj:`None`)
        binary (bool): Whether to binarize the weights on the edges to 0 and 1. 
            If set to True, binarization are done after topk and threshold, 
            the edge weights that still have values are set to 1, 
            otherwise they are set to 0. (default: :obj:`False`)
        complete_graph (bool): Whether to build as a complete graph. If False, 
            only construct edges between electrodes based on non-zero elements; 
            if True, construct variables between all electrodes and set the 
            weight of non-existing edges to 0. (default: :obj:`False`)
    
    .. automethod:: __call__
    '''
    def __init__(self,
                 adj: List[List],
                 add_self_loop: bool = True,
                 threshold: Union[float, None] = None,
                 top_k: Union[int, None] = None,
                 binary: bool = False,
                 complete_graph: bool = False):
        super(ToG, self).__init__()

        self.add_self_loop = add_self_loop
        self.threshold = threshold
        self.top_k = top_k
        self.binary = binary
        self.complete_graph = complete_graph

        adj = torch.tensor(adj).float()

        if add_self_loop:
            adj = adj + torch.eye(adj.shape[0])

        if not self.threshold is None:
            adj[adj < self.threshold] = 0

        if not self.top_k is None:
            rows = []
            for row in adj:
                vals, index = row.topk(self.top_k)
                topk = torch.zeros_like(row)
                topk[index] = vals
                rows.append(topk)
            adj = torch.stack(rows)

        if self.binary:
            adj[adj != 0] = 1.0

        if self.complete_graph:
            adj[adj == 0] = 1e-6

        self.adj = adj.to_sparse()

    def __call__(self,
                 eeg: Union[np.ndarray, torch.Tensor],
                 **kwargs) -> Data:
        r'''
        Args:
            eeg (np.ndarray): The input EEG signals in shape of 
                [number of electrodes, number of data points].
            
        Returns:
            torch_geometric.data.Data: The graph representation data types 
                that torch_geometric can accept. Nodes correspond to electrodes, 
                and edges are determined via the given adjacency matrix.
        '''
        data = Data(edge_index=self.adj._indices())
        if isinstance(eeg, np.ndarray):
            data.x = torch.from_numpy(eeg).float()
        else:
            data.x = eeg
        data.edge_attr = self.adj._values()

        return data


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, epochs, targets, transforms=None):
        super(EEGDataset, self).__init__()
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, idx):
        return self.epochs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
    

class MaskedEEGDataset(torch.utils.data.Dataset):
    # Channel-Masked EEG Dataset
    # epochs: n_epochs of (n_samples, n_channels_i)
    # targets: n_epochs of (1)
    # chlabels: list of channel labels (n_channels_i)
    # full_chlabels: list of full channel labels (n_channels)
    # channel_mask: list of channel masks (n_channels)
    def __init__(
            self, epochs, targets, transforms=None, chlabels=None, full_chlabels=None, 
        ):
        super(MaskedEEGDataset, self).__init__()
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.targets = torch.LongTensor(targets)

        self.chlabels = chlabels
        self.full_chlabels = full_chlabels
        self.channel_mask = torch.zeros(len(self.full_chlabels), dtype=torch.bool)
        for i, ch in enumerate(self.chlabels):
            if ch in self.full_chlabels:
                ch_idx = full_chlabels.index(ch)
                self.channel_mask[ch_idx] = True
            else:
                print(f"Warning: channel {ch} not found in full channel list")

        used_chlabels = [self.full_chlabels[i] for i in range(len(self.channel_mask)) if self.channel_mask[i]]
        print(f"Using {len(used_chlabels)}/{len(self.full_chlabels)} channels: {used_chlabels}")

    def __getitem__(self, idx):
        epoch = self.epochs[idx]
        n_channels = len(self.full_chlabels)
        # allow for different number of channels in each epoch
        epoch_full = torch.zeros(*epoch.shape[:-1], n_channels)
        ch_idx = 0
        for i, ch in enumerate(self.channel_mask):
            if ch:
                epoch_full[..., i] = epoch[..., ch_idx]
                ch_idx += 1
        return epoch_full, self.targets[idx], self.channel_mask

    def __len__(self):
        return len(self.targets)
    

class MaskedGraphEEGDataset(torch.utils.data.Dataset):
    # Channel-Masked Graph EEG Dataset
    # epochs: list of EEG epochs n_epochs of (n_samples, n_channels_i)
    # targets: list of target labels (n_epochs)
    # chlabels: list of channel labels (n_channels_i)
    # full_chlabels: list of full channel labels (n_channels)
    # full_adj: full adjacency matrix (n_channels x n_channels)
    def __init__(
            self, epochs, targets, transforms=None, chlabels=None, full_chlabels=None, full_adj=None, 
        ):
        super(MaskedGraphEEGDataset, self).__init__()
        tf_ToG = ToG(adj=full_adj)
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.epochs = [tf_ToG(eeg=epoch) for epoch in self.epochs]
        self.targets = torch.LongTensor(targets)

        self.chlabels = chlabels
        self.full_chlabels = full_chlabels
        self.channel_mask = torch.zeros(len(self.full_chlabels), dtype=torch.bool)
        for i, ch in enumerate(self.chlabels):
            if ch in self.full_chlabels:
                ch_idx = full_chlabels.index(ch)
                self.channel_mask[ch_idx] = True
            else:
                print(f"Warning: channel {ch} not found in full channel list")

        used_chlabels = [self.full_chlabels[i] for i in range(len(self.channel_mask)) if self.channel_mask[i]]
        print(f"Using {len(used_chlabels)}/{len(self.full_chlabels)} channels: {used_chlabels}")

    def __getitem__(self, idx):
        data = self.epochs[idx]
        epoch = data.x
        n_channels = len(self.full_chlabels)
        # allow for different number of channels in each epoch
        epoch_full = torch.zeros(*epoch.shape[:-1], n_channels)
        ch_idx = 0
        for i, ch in enumerate(self.channel_mask):
            if ch:
                epoch_full[..., i] = epoch[..., ch_idx]
                ch_idx += 1
        data.x = epoch_full.transpose(-1, -2) # (n_samples, n_channels, n_timepoints)
        data.channel_mask = self.channel_mask
        data.y = self.targets[idx]
        return data

    def __len__(self):
        return len(self.targets)
    

if __name__ == '__main__':
    
    from torch_geometric.loader import DataLoader

    x = torch.rand(10, 256, 6)
    y = torch.randint(0, 4, (10,))
    dataset = MaskedGraphEEGDataset(
        x, y, 
        chlabels=['F3', 'F4', 'C3', 'C4', 'O1', 'O2'], 
        full_chlabels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'], 
        full_adj=np.eye(10),
    )
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for data in loader:
        print(data)
        break
>>>>>>> de32b1275879a7d77f6de917d6521e87e2f591e4
    