
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric import nn as pyg_nn
from torch_geometric.nn import BatchNorm, GraphNorm, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import gnn


class ConvPooling(nn.Module):
    def __init__(self, n_channels=32, embed_dim=64, droprate=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=n_channels, groups=embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
    def forward(self, x):
        # x: (B, N, P)
        # output: (B, P)
        x = x.permute(0, 2, 1) # (B, P, N)
        x = self.conv(x)
        x = x.squeeze(2) # (B, P)
        return x
    

class MixerPooling(nn.Module):
    def __init__(self, embed_dim, ratio=1, droprate=0., pooling=True):
        super().__init__()
        self.pooling = pooling
        inner_dim = int(embed_dim * ratio)
        # channel mixing
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, inner_dim, kernel_size=1, groups=embed_dim, bias=False),
            nn.BatchNorm1d(inner_dim),
            nn.GELU(),
            nn.Dropout(droprate),
            nn.Conv1d(inner_dim, embed_dim, kernel_size=1, bias=False),
            nn.Dropout(droprate),
        )

    def forward(self, x):
        # x: (B, N, P)
        # output: (B, N, P) or (B, N*P) determined by flatten
        x = self.conv(x)
        if self.pooling:
            # mean pooling on channels
            x = x.mean(dim=1)
        else:
            x = x.flatten(1)  # (B, N*P)
        return x
    
        
class GNNPooling(torch.nn.Module):
    def __init__(
            self, n_channels, input_dim, embed_dim=128, n_layers=3, 
            droprate=0., pooling=True, gnn_type='gcn',
            adj=None, threshold=0.5, 
        ):
        super().__init__()
        assert n_layers >= 1, "n_layers must be >= 1"
        self.pooling = pooling
        if gnn_type == 'gcn':
            gnn_layer = gnn.GCNConv
        elif gnn_type == 'gat':
            gnn_layer = gnn.GATConv
        elif gnn_type == 'cheb':
            gnn_layer = gnn.ChebConv
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        if gnn_type == 'cheb':
            self.conv1 = gnn_layer(input_dim, embed_dim, K=5, bias=False)
        else:
            self.conv1 = gnn_layer(input_dim, embed_dim, bias=False)
        self.norm1 = BatchNorm4T(embed_dim)
        # self.norm1 = nn.LayerNorm(embed_dim) # layer norm does not work well
        self.proj_res = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers - 1):
            if gnn_type == 'cheb':
                self.convs.append(gnn_layer(embed_dim, embed_dim, K=3, bias=False))
            else:
                self.convs.append(gnn_layer(embed_dim, embed_dim, bias=False))
            norm_i = BatchNorm4T(embed_dim)
            # norm_i = nn.LayerNorm(embed_dim) # layer norm does not work well
            self.norms.append(norm_i)
        self.dropout = nn.Dropout(droprate)

        if adj is None:
            dist_matrix = torch.ones((n_channels, n_channels)) - torch.eye(n_channels)
        else:
            if isinstance(adj, dict): # 2D coordinate list
                dist_matrix = dist_from_xy(adj)
            elif isinstance(adj, tuple) or isinstance(adj, list): # position list
                dist_matrix = dist_from_xyz(adj).float()
            else: # adjacency matrix
                dist_matrix = dist_from_adj(adj).float()
        adj_dist = torch.exp(-dist_matrix / dist_matrix.std())
        adj_dist[adj_dist < threshold] = 0
        self.register_buffer('adj_dist', adj_dist)
        self.adj_learn = nn.Parameter(torch.FloatTensor(n_channels, n_channels))
        self.alphas = nn.Parameter(torch.ones(n_layers))

        # self.convs[-1].register_forward_hook(self.hook_fn)

    def forward(self, x, channel_mask=None):
        # x: (B, N, P)
        # output: (B, P) or (B, N, P) determined by pooling
        batch_size = x.size(0)
        
        adj = self.alphas[0] * self.adj_dist + (1-self.alphas[0]) * self.adj_learn
        adj_norm = gnn.normalize_A(adj, laplacian=False)
        x = F.relu(self.norm1(self.conv1(x, adj_norm))) # + self.proj_res(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            adj = self.alphas[i+1] * self.adj_dist + (1-self.alphas[i+1]) * self.adj_learn
            adj_norm = gnn.normalize_A(adj, laplacian=False)
            x = F.relu(norm(conv(x, adj_norm))) # + x # residual connection

        if self.pooling:
            x = x.mean(dim=1)  # (B, P)
        else:
            x = x.view(batch_size, -1, x.shape[-1])  # (B, N, P)

        x = self.dropout(x)
        return x
    
    def hook_fn(self, m, i, o):
        print("mean:", o.mean().item(), "std:", o.std().item())
    

class GNNPooling_pyg(torch.nn.Module):
    def __init__(
            self, n_channels, input_dim, embed_dim=128, n_layers=3, 
            droprate=0., pooling=True, gnn_type='cheb', norm='batch',
            adj=None, threshold=0.5, 
        ):
        super().__init__()
        assert n_layers >= 1, "n_layers must be >= 1"
        self.pooling = pooling
        if gnn_type == 'gcn':
            gnn_layer = pyg_nn.GCNConv
        elif gnn_type == 'gat':
            gnn_layer = pyg_nn.GATConv
        elif gnn_type == 'cheb':
            gnn_layer = pyg_nn.ChebConv
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        if norm == 'none':
            norm_layer = None
        elif norm == 'batch':
            norm_layer = BatchNorm # 目前来看 batch norm 效果最好
        elif norm == 'graph':
            norm_layer = GraphNorm
        elif norm == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        
        if gnn_type == 'cheb':
            self.conv1 = gnn_layer(input_dim, embed_dim, K=5, bias=False)
        else:
            self.conv1 = gnn_layer(input_dim, embed_dim, bias=False)
        self.norm1 = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(n_layers - 1):
            if gnn_type == 'cheb':
                self.convs.append(gnn_layer(embed_dim, embed_dim, K=3, bias=False))
            else:
                self.convs.append(gnn_layer(embed_dim, embed_dim, bias=False))
            norm_i = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
            self.norms.append(norm_i)
        self.dropout = nn.Dropout(droprate)

        if adj is None:
            dist_matrix = torch.ones((n_channels, n_channels)) - torch.eye(n_channels)
        else:
            if isinstance(adj, dict): # 2D coordinate list
                dist_matrix = dist_from_xy(adj)
            elif isinstance(adj, tuple) or isinstance(adj, list): # position list
                dist_matrix = dist_from_xyz(adj).float()
            else: # adjacency matrix
                dist_matrix = dist_from_adj(adj).float()
        adj_dist = torch.exp(-dist_matrix / dist_matrix.std())
        # edge_index, edge_weight = dense_to_sparse(adj_dist)
        edge_index = torch.nonzero(adj_dist > threshold).t()
        edge_weight = adj_dist[adj_dist > threshold].unsqueeze(1)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        # self.convs[-1].register_forward_hook(self.hook_fn)

    def forward(self, x, channel_mask=None):
        # x: (batch_size, n_nodes, hidden_dim)
        if isinstance(x, Data) or isinstance(x, Batch):
            batch = x
        else:
            # convert x to graph batch
            batch = self.create_graph_batch(x, self.edge_index, self.edge_weight, channel_mask)
        x, edge_index, edge_weight, batch_index, batch_size = \
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.batch_size

        if hasattr(batch, 'channel_mask'):
            channel_mask = batch.channel_mask  # (n_nodes,)
            mask = channel_mask[edge_index[0]] & channel_mask[edge_index[1]]
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]

        # 图卷积层
        x = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight)))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index, edge_weight))) # + x # residual connection
        # 全局池化
        if channel_mask is not None:
            x = x[channel_mask]
            batch_index = batch_index[channel_mask]
        
        if self.pooling:
            x = global_mean_pool(x, batch_index)
        else:
            x = x.view(batch_size, -1, x.shape[-1])  # (B, N, P)

        out = self.dropout(x)
        return out
    
    def hook_fn(self, m, i, o):
        print("mean:", o.mean().item(), "std:", o.std().item())
    
    def create_graph_batch(self, x, edge_index, edge_weight, channel_mask=None):
        """
        x: (batch_size, n_channels, n_features)
        edge_index: [2, num_edges], 全体样本共享
        edge_weight: [num_edges, 1], 全体样本共享
        """
        data_list = []
        for i in range(x.size(0)):
            if channel_mask is None:
                data = Data(x=x[i], edge_index=edge_index, edge_attr=edge_weight)
            else:
                mask = channel_mask[i]
                data = Data(x=x[i], edge_index=edge_index, edge_attr=edge_weight, channel_mask=mask)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        return batch
    

class AttnPooling(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers=2, n_heads=4, droprate=0.1, 
                 pooling=True, use_conv=False, use_residual=False, adj=None):
        super().__init__()
        self.pooling = pooling
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.use_conv = use_conv
        self.use_residual = use_residual

        if adj is None:
            dist_matrix = torch.ones((n_channels, n_channels)) - torch.eye(n_channels)
        else:
            if isinstance(adj, dict): # 2D coordinate list
                dist_matrix = dist_from_xy(adj)
            elif isinstance(adj, tuple) or isinstance(adj, list): # position list
                dist_matrix = dist_from_xyz(adj).float()
            else: # adjacency matrix
                dist_matrix = dist_from_adj(adj).float()

        # create attention bias from distance matrix, refer to RetNet
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/8), math.log(1/64), n_heads)))
        attn_bias = []
        for i in range(self.n_heads):
            attn_bias_i = self._get_attn_bias(self.gammas[i], dist_matrix)
            attn_bias.append(attn_bias_i)
        attn_bias = torch.stack(attn_bias, dim=0)  # [h, q_len, k_len]
        self.register_buffer('attn_bias', attn_bias)

        # 多头自注意力（用于通道间融合）
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(MultiheadAttention(embed_dim, n_heads, dropout=droprate))
        self.norm = BatchNorm4T(embed_dim) # nn.LayerNorm(embed_dim)

        # 可选：Conv1d 融合（通道间 depthwise 聚合）
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=n_channels, groups=embed_dim, bias=False),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(droprate)
            )

    def forward(self, x, channel_mask=None):
        """ x: (B, N, F) """
        B, N, _ = x.shape
        # create mask for cls token and channel mask
        if channel_mask is None:
            channel_mask = torch.ones(B, N, dtype=bool).to(x.device)
        mask = channel_mask # (bs, seq_length)
        attn_bias = self.attn_bias.unsqueeze(0).expand(x.size(0), -1, -1, -1).to(x.device)  # [b, h, q_len, k_len]

        out = x
        attn_weights = []
        for layer in self.layers:
            attn_out, attn_w = layer(out, out, out, mask=~mask, attn_bias=attn_bias)
            out = out + attn_out # residual
            attn_weights.append(attn_w)

        # Conv1d 作为增强路径
        if self.use_conv:
            x_conv = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, F)
            out = out + x_conv

        out = self.norm(out)
        if self.pooling:
            out = out.mean(dim=1)
        else:
            out = out.flatten(1)

        return out, attn_weights
    
    def _get_attn_bias(self, gamma, dist_matrix):
        D = gamma ** dist_matrix
        D[dist_matrix == torch.inf] = 0
        return D
    

class GraphormerPooling(nn.Module):
    def __init__(
            self, n_channels, embed_dim=128, n_layers=2, n_heads=4, pooling=True, 
            droprate_input=0., droprate_fc=0., droprate_attn=0., adj=None,
        ):
        super().__init__()
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout_input = nn.Dropout(droprate_input)
        # 
        if adj is None:
            dist_matrix = torch.ones((n_channels, n_channels)) - torch.eye(n_channels)
        else:
            if isinstance(adj, dict): # 2D coordinate list
                dist_matrix = dist_from_xy(adj)
            elif isinstance(adj, tuple) or isinstance(adj, list): # position list
                dist_matrix = dist_from_xyz(adj).float()
            else: # adjacency matrix
                dist_matrix = dist_from_adj(adj).float()
        
        # create attention bias from distance matrix, refer to RetNet
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/8), math.log(1/64), n_heads)))
        attn_bias = []
        for i in range(self.n_heads):
            attn_bias_i = self._get_attn_bias(self.gammas[i], dist_matrix)
            attn_bias.append(attn_bias_i)
        attn_bias = torch.stack(attn_bias, dim=0)  # [h, q_len, k_len]
        self.register_buffer('attn_bias', attn_bias)

        # virtual node
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoders = nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, embed_dim * 4, droprate_fc, droprate_attn, bias_type='dot')
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(embed_dim)

        self.init_weights(self.encoders, n_layers)

    def init_weights(self, module, n_layers):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, x, channel_mask=None):
        # x: (batch_size, n_nodes, embed_dim)
        if isinstance(x, Data) or isinstance(x, Batch):
            x, edge_index, edge_weight, batch_index, batch_size = \
                x.x, x.edge_index, x.edge_attr, x.batch, x.batch_size
            n_nodes = x.size(0) // batch_size
            x = x.reshape(batch_size, n_nodes, -1) # (batch_size, n_nodes, embed_dim)
        batch_size, n_nodes, _ = x.shape
        # append cls token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, seq_length+1, embed_dim)
        # create mask for cls token and channel mask
        if channel_mask is None:
            channel_mask = torch.ones(batch_size, n_nodes, dtype=bool).to(x.device)
        mask_token = torch.zeros(x.size(0), 1, dtype=torch.bool).to(x.device)
        mask = torch.cat((mask_token, channel_mask), dim=1) # (bs, seq_length+1)
        # create distance matrix from shortest paths
        attn_bias_ext = torch.zeros((self.n_heads, x.size(1), x.size(1)), dtype=torch.long)
        attn_bias_ext[:, 1:, 1:] = self.attn_bias
        attn_bias_ext = attn_bias_ext.unsqueeze(0).expand(x.size(0), -1, -1, -1).to(x.device)  # [b, h, q_len, k_len]
        ## visualize the attention bias
        # import matplotlib.pyplot as plt
        # im1 = plt.imshow(dist_matrix.detach().cpu().numpy())
        # plt.colorbar(mappable=im1)
        # plt.show()

        x = self.dropout_input(x)
        out = x
        for encoder in self.encoders:
            attn_out, attn_weights = encoder(out, mask=~mask, attn_bias=attn_bias_ext)
        out = self.ln(attn_out)
        if self.pooling:
            out = out[:, 0] # only use the first node (virtual node)
        else:
            out = out[:, 1:].flatten(1) # (bs, n_nodes*embed_dim)

        return out
    
    def _get_attn_bias(self, gamma, dist_matrix):
        D = gamma ** dist_matrix
        D[dist_matrix == torch.inf] = 0
        return D
    
def dist_from_xy(coordinates_map):
    N = len(coordinates_map)
    dist = torch.zeros((N, N), dtype=torch.float32)
    for i, (x1, y1) in enumerate(coordinates_map.values()):
        for j, (x2, y2) in enumerate(coordinates_map.values()):
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dist[i, j] = round(distance, 3)
    return dist

def dist_from_xyz(xyz):
    """
    :param xyz: (N, 3), xyz coordinates of nodes
    :return: distance matrix (N, N)
    """
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.tensor(xyz, dtype=torch.float32)
    N = xyz.shape[0]
    dist = torch.zeros((N, N), dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            dist[i, j] = torch.norm(xyz[i] - xyz[j])
    return dist

def dist_from_adj(adj, max_distance=8):
    """ 
    Compute the distance matrix from the adjacency matrix using Floyd-Warshall algorithm.
    :param adj: adjacency matrix (N, N)
    :param max_distance: the maximum distance between two nodes, default: n_spatial_relations - 1
    :return: distance matrix (N, N)
    """
    N = adj.shape[0]
    dist = torch.full((N, N), float('inf'))
    dist[adj == 1] = 1
    dist[torch.arange(N), torch.arange(N)] = 0
    for k in range(N):
        dist = torch.min(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))
    dist[torch.isinf(dist)] = max_distance  # 将无穷距离替换为最大距离
    dist = dist.clamp(0, max_distance).to(torch.long)
    return dist

def dist_from_edge_index(edge_index, n_nodes, batch_size):
    # 创建单个图的邻接矩阵
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)
    # 只使用前 num_edges 个索引
    num_edges = edge_index.size(1) // batch_size
    adj[edge_index[0, :num_edges], edge_index[1, :num_edges]] = True
    return dist_from_adj(adj)

def attention(query, key, value, mask=None, droprate=None, attn_bias=None, bias_type='dot'):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'),)
    if bias_type == 'dot':
        p_attn = F.softmax(scores, dim=-1)
        # The order is different from Gradformer, which applies attn_bias before softmax
        if attn_bias is not None:
            p_attn = p_attn * attn_bias
    elif bias_type == 'add':
        if attn_bias is not None:
            scores += attn_bias
        p_attn = F.softmax(scores, dim=-1)
    if droprate is not None:
        p_attn = droprate(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0., dropout_proj=0.):
        super(MultiheadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_qkv = d_qkv = d_model // n_heads
        self.scale = d_qkv ** -0.5

        self.W_Q = nn.Linear(d_model, d_qkv * n_heads)
        self.W_K = nn.Linear(d_model, d_qkv * n_heads)
        self.W_V = nn.Linear(d_model, d_qkv * n_heads)
        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Sequential(
            nn.Linear(n_heads * d_qkv, d_model), nn.Dropout(dropout_proj))

    def forward(self, q, k, v, mask=None, attn_bias=None, bias_type='dot'):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)

        d_k = self.d_qkv
        d_v = self.d_qkv
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.W_Q(q).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2) # [b, h, q_len, d_k]
        k = self.W_K(k).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2) # [b, h, v_len, d_v]
        v = self.W_V(v).view(batch_size, -1, self.n_heads, d_v).transpose(1, 2) # [b, h, d_k, k_len]

        # Rettention
        output, attn_weights = attention(
            q, k, v, mask=mask, droprate=self.dropout, attn_bias=attn_bias, bias_type=bias_type)

        # output: [bs x n_heads x q_len x d_v], 
        # attn: [bs x n_heads x q_len x q_len], 
        # scores: [bs x n_heads x max_q_len x q_len]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_qkv)

        output = self.out_proj(output)

        return output, attn_weights
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, droprate=0., attn_droprate=0., bias_type='dot'):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.bias_type = bias_type

        self.attn = MultiheadAttention(d_model, n_heads, dropout=attn_droprate)
        self.attn_ln = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(droprate),
            nn.Linear(d_ff, d_model)
        )
        self.ff_ln = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, attn_bias=None):
        # 1. self attention
        x = self.attn_ln(x)
        attn_out, attn_weights = self.attn(x, x, x, mask=mask, attn_bias=attn_bias, bias_type=self.bias_type)
        x = x + attn_out
        # 2. ff network
        x = x + self.ff(self.ff_ln(x))
        return x, attn_weights.detach()

class BatchNorm4T(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.bn(x)
        x = x.transpose(1,2)
        return x
    

if __name__ == '__main__':

    x = torch.randn(5, 32, 128)
    adj = torch.randn(32, 32).numpy()
    model = GNNPooling(
        n_channels=32, input_dim=128, embed_dim=64, n_layers=3, 
        droprate=0.1, pooling=False, gnn_type='gcn', 
        adj=adj, threshold=0.5,
    )
    y = model(x)
    print(y.shape)
    