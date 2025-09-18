<<<<<<< HEAD

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebNet(nn.Module):
    def __init__(self, n_classes, in_dim, out_dim, n_layers=3, adj=None, n_nodes=None, K=3, dropout=0.5):
        super(ChebNet, self).__init__()
        if adj is not None:
            self.adj = adj
        else:
            self.adj = nn.Parameter(torch.FloatTensor(n_nodes, n_nodes))

        self.cheb_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.cheb_layers.append(ChebConv(in_dim, out_dim, K))
            else:
                self.cheb_layers.append(ChebConv(out_dim, out_dim, K))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, n_classes) if n_classes > 0 else nn.Identity()

    def forward(self, x, L=None):
        if L is None:
            L = normalize_A(self.adj)
        for layer in self.cheb_layers:
            x = layer(x, L)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K, bias=True):
        super(ChebConv, self).__init__()
        self.K = K
        self.gcs = nn.ModuleList()   #https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gcs.append(GCNConv(in_dim, out_dim, bias=bias))

    def forward(self, x, L):
        adj = self.generate_cheb_adj(L, self.K, x.device)
        for i in range(len(self.gcs)):
            if i == 0:
                result = self.gcs[i](x, adj[i])
            else:
                result += self.gcs[i](x, adj[i])
        result = F.relu(result)
        return result
    
    def generate_cheb_adj(self, L, K, device):
        # L: (n_nodes, n_nodes)
        # K: number of Chebyshev polynomials
        N = L.size(0)
        adj = [torch.eye(N).to(device)]
        for k in range(1, K):
            adj_k = torch.matmul(adj[k-1], L) # first-order Chebyshev polynomial
            # adj_k = 2 * torch.matmul(L, adj[k-1]) - adj[k-2] # second-order
            adj.append(adj_k)
        return adj
    

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.0, alpha=0.2, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))        # (H, F_in, F_out)
        self.attn_l = nn.Parameter(torch.Tensor(heads, out_dim))           # (H, F_out)
        self.attn_r = nn.Parameter(torch.Tensor(heads, out_dim))           # (H, F_out)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        if bias:
            if concat:
                self.bias = nn.Parameter(torch.Tensor(heads * out_dim))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: (B, N, F_in)
        adj: (B, N, N)
        """
        B, N, _ = x.size()

        # 1. Linear projection: (B, H, N, F_out)
        x_proj = torch.einsum('bnf,hfo->bhno', x, self.W)  # x: (B, N, F_in), W: (H, F_in, F_out)

        # 2. Compute attention logits
        attn_l = torch.einsum('bhnd,hd->bhn', x_proj, self.attn_l)  # (B, H, N)
        attn_r = torch.einsum('bhnd,hd->bhn', x_proj, self.attn_r)  # (B, H, N)
        e = self.leakyrelu(attn_l.unsqueeze(-1) + attn_r.unsqueeze(-2))  # (B, H, N, N)

        # 3. Masked attention
        mask = (adj > 0).unsqueeze(0).unsqueeze(0).expand(B, self.heads, N, N) # (B, H, N, N)
        e = e.masked_fill(~mask, float("-inf"))                          # (B, H, N, N)
        alpha = F.softmax(e, dim=-1)                                     # (B, H, N, N)
        alpha = self.dropout(alpha)

        # 4. Attention-weighted feature aggregation
        out = torch.matmul(alpha, x_proj)  # (B, H, N, F_out)

        # 5. Merge heads
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_dim)  # (B, N, H * F_out)
        else:
            out = out.mean(dim=1)  # (B, N, F_out)

        # 6. Bias + activation
        if self.bias is not None:
            out = out + self.bias
        return F.elu(out)


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)    
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out
        
    def forward1(self, x, adj): # LGGNet
        output = torch.matmul(x, self.weight)-self.bias
        output = torch.matmul(adj, output)
        return output
        

def normalize_A(A, add_self_loop=True, symmetry=True, laplacian=False, eps=1e-10):
    """
    1. 对称归一化邻接矩阵 A，用于 GCNConv：
    A_norm = D^{-1/2} (A + I) D^{-1/2}

    2. 缩放后的对称拉普拉斯矩阵，用于 ChebConv：
    L = I - D^{-1/2} A D^{-1/2}
    L_scaled = 2L / lambda_max - I
    """
    A = F.relu(A) # ReLU activation to ensure non-negativity

    if add_self_loop:
        A = A + torch.eye(A.size(0)).to(A.device)
    
    if symmetry:
        A = (A + A.T) / 2
        
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt(d + eps)
    D = torch.diag_embed(d)
    L = torch.matmul(torch.matmul(D, A), D)

    if laplacian:
        L = torch.eye(A.size(0)).to(A.device) - L
        lambda_max = torch.linalg.eigvalsh(L).max().real.detach()
        L = 2.0 * L / lambda_max - torch.eye(A.size(0)).to(A.device)
    return L
        

if __name__ == '__main__':

    x = torch.randn(16, 100, 10)
    adj = torch.randn(100, 100)

    gcn = GCNConv(10, 20)
    output = gcn(x, adj)
    print(output.shape)

    gat = GATConv(10, 20, heads=4)
    output = gat(x, adj)
    print(output.shape)

    # model = ChebNet(5, 10, 20, 2, adj=adj)
    model = ChebNet(5, 10, 20, 2, n_nodes=100)
    output = model(x)
    print(output.shape)
=======

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebNet(nn.Module):
    def __init__(self, n_classes, in_dim, out_dim, n_layers=3, adj=None, n_nodes=None, K=3, dropout=0.5):
        super(ChebNet, self).__init__()
        if adj is not None:
            self.adj = adj
        else:
            self.adj = nn.Parameter(torch.FloatTensor(n_nodes, n_nodes))

        self.cheb_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.cheb_layers.append(ChebConv(in_dim, out_dim, K))
            else:
                self.cheb_layers.append(ChebConv(out_dim, out_dim, K))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, n_classes) if n_classes > 0 else nn.Identity()

    def forward(self, x, L=None):
        if L is None:
            L = normalize_A(self.adj)
        for layer in self.cheb_layers:
            x = layer(x, L)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K, bias=True):
        super(ChebConv, self).__init__()
        self.K = K
        self.gcs = nn.ModuleList()   #https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gcs.append(GCNConv(in_dim, out_dim, bias=bias))

    def forward(self, x, L):
        adj = self.generate_cheb_adj(L, self.K, x.device)
        for i in range(len(self.gcs)):
            if i == 0:
                result = self.gcs[i](x, adj[i])
            else:
                result += self.gcs[i](x, adj[i])
        result = F.relu(result)
        return result
    
    def generate_cheb_adj(self, L, K, device):
        # L: (n_nodes, n_nodes)
        # K: number of Chebyshev polynomials
        N = L.size(0)
        adj = [torch.eye(N).to(device)]
        for k in range(1, K):
            adj_k = torch.matmul(adj[k-1], L) # first-order Chebyshev polynomial
            # adj_k = 2 * torch.matmul(L, adj[k-1]) - adj[k-2] # second-order
            adj.append(adj_k)
        return adj
    

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.0, alpha=0.2, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))        # (H, F_in, F_out)
        self.attn_l = nn.Parameter(torch.Tensor(heads, out_dim))           # (H, F_out)
        self.attn_r = nn.Parameter(torch.Tensor(heads, out_dim))           # (H, F_out)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        if bias:
            if concat:
                self.bias = nn.Parameter(torch.Tensor(heads * out_dim))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: (B, N, F_in)
        adj: (B, N, N)
        """
        B, N, _ = x.size()

        # 1. Linear projection: (B, H, N, F_out)
        x_proj = torch.einsum('bnf,hfo->bhno', x, self.W)  # x: (B, N, F_in), W: (H, F_in, F_out)

        # 2. Compute attention logits
        attn_l = torch.einsum('bhnd,hd->bhn', x_proj, self.attn_l)  # (B, H, N)
        attn_r = torch.einsum('bhnd,hd->bhn', x_proj, self.attn_r)  # (B, H, N)
        e = self.leakyrelu(attn_l.unsqueeze(-1) + attn_r.unsqueeze(-2))  # (B, H, N, N)

        # 3. Masked attention
        mask = (adj > 0).unsqueeze(0).unsqueeze(0).expand(B, self.heads, N, N) # (B, H, N, N)
        e = e.masked_fill(~mask, float("-inf"))                          # (B, H, N, N)
        alpha = F.softmax(e, dim=-1)                                     # (B, H, N, N)
        alpha = self.dropout(alpha)

        # 4. Attention-weighted feature aggregation
        out = torch.matmul(alpha, x_proj)  # (B, H, N, F_out)

        # 5. Merge heads
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.out_dim)  # (B, N, H * F_out)
        else:
            out = out.mean(dim=1)  # (B, N, F_out)

        # 6. Bias + activation
        if self.bias is not None:
            out = out + self.bias
        return F.elu(out)


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)    
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out
        
    def forward1(self, x, adj): # LGGNet
        output = torch.matmul(x, self.weight)-self.bias
        output = torch.matmul(adj, output)
        return output
        

def normalize_A(A, add_self_loop=True, symmetry=True, laplacian=False, eps=1e-10):
    """
    1. 对称归一化邻接矩阵 A，用于 GCNConv：
    A_norm = D^{-1/2} (A + I) D^{-1/2}

    2. 缩放后的对称拉普拉斯矩阵，用于 ChebConv：
    L = I - D^{-1/2} A D^{-1/2}
    L_scaled = 2L / lambda_max - I
    """
    A = F.relu(A) # ReLU activation to ensure non-negativity

    if add_self_loop:
        A = A + torch.eye(A.size(0)).to(A.device)
    
    if symmetry:
        A = (A + A.T) / 2
        
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt(d + eps)
    D = torch.diag_embed(d)
    L = torch.matmul(torch.matmul(D, A), D)

    if laplacian:
        L = torch.eye(A.size(0)).to(A.device) - L
        lambda_max = torch.linalg.eigvalsh(L).max().real.detach()
        L = 2.0 * L / lambda_max - torch.eye(A.size(0)).to(A.device)
    return L
        

if __name__ == '__main__':

    x = torch.randn(16, 100, 10)
    adj = torch.randn(100, 100)

    gcn = GCNConv(10, 20)
    output = gcn(x, adj)
    print(output.shape)

    gat = GATConv(10, 20, heads=4)
    output = gat(x, adj)
    print(output.shape)

    # model = ChebNet(5, 10, 20, 2, adj=adj)
    model = ChebNet(5, 10, 20, 2, n_nodes=100)
    output = model(x)
    print(output.shape)
>>>>>>> de32b1275879a7d77f6de917d6521e87e2f591e4
