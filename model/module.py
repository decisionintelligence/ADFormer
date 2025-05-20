import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

    
class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """
    def __init__(self, d_model, swiglu_ratio=8/3):
        """
        Args:
            d_model (int): Dimension of the input features.
        """
        super().__init__()
        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = nn.Linear(d_model, int(d_model * swiglu_ratio))
        self.W1 = nn.Linear(d_model, int(d_model * swiglu_ratio))
        self.W2 = nn.Linear(int(d_model * swiglu_ratio), d_model)
    
    def forward(self, x):
        """
        Forward pass for SwiGLU.
        
        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).
        
        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        # Apply the gates
        g = F.silu(self.WG(x))  # Activation part
        z = self.W1(x)            # Linear part
        # Element-wise multiplication and projection
        return self.W2(g * z)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()
    

class SpatialPE(nn.Module):
    def __init__(self, SE_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(SE_dim, embed_dim)

    def forward(self, spa_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(spa_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc
    

class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, SE_dim,
        add_time_in_day=True, add_day_in_week=True, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spatial_embedding = SpatialPE(SE_dim, embed_dim)
        

    def forward(self, x, spa_mx=None):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)

        if self.add_time_in_day:
            de = self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
            x += de
        if self.add_day_in_week:
            we = self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
            x += we
        
        if spa_mx is not None:
            x += self.spatial_embedding(spa_mx)
        return x
    

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


    
def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))


def cluster_regions(distance_matrix, target_clusters, balance, tolerance):
    """
    Cluster regions to reach a target number of clusters.

    :param distance_matrix: A 263x263 distance matrix (numpy array)
    :param target_clusters: The desired number of clusters after merging (int)
    :return: A dictionary where each key is the new cluster index (starting from 1),
             and each value is a list of original region indices (starting from 0) contained in that cluster.
    """

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square (NxN)!")
    
    condensed_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    linkage_matrix = linkage(condensed_dist, method='average')
    labels = fcluster(linkage_matrix, target_clusters, criterion='maxclust')  # labels表示对应点所属于的聚簇

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    if not balance:
        return [points for points in clusters.values()]
    
    avg_size = len(distance_matrix) // target_clusters
    max_size = int(avg_size * (1 + tolerance))
    
    # Adjust clusters that are too large
    for label in list(clusters.keys()):
        while len(clusters[label]) > max_size:
            # Remove a point from the oversized cluster
            point_to_move = clusters[label].pop()
            
            # Find the nearest smaller cluster to reassign the point
            min_label = None
            min_distance = float('inf')
            for other_label, other_points in clusters.items():
                if other_label != label and len(other_points) < max_size:
                    # Compute the average distance from the point to the other cluster
                    avg_dist = np.mean([distance_matrix[point_to_move, p] for p in other_points])
                    if avg_dist < min_distance:
                        min_distance = avg_dist
                        min_label = other_label
            
            if min_label is not None:
                clusters[min_label].append(point_to_move)

    balanced_clusters = [points for points in clusters.values()]
    return balanced_clusters




def hierarchical_clustering(distance_matrix, cluster_targets, balance, tolerance):
    """
    Perform hierarchical clustering by progressively merging clusters 
    based on the distance matrix until reaching each target number of clusters.
    
    :param distance_matrix: Initial distance matrix (numpy array)
    :param cluster_targets: List of target numbers of clusters (list of int)
    :return: A list of clustering results for each stage, 
             where each result is a dictionary.
    """

    results = []
    current_matrix = distance_matrix
    
    for target in cluster_targets:
        clusters = cluster_regions(current_matrix, target, balance=balance, tolerance=tolerance)
        results.append(clusters)
        
        # Recalculate the distance matrix for the new clusters
        new_matrix = np.zeros((target, target))
        for i, group_i in enumerate(clusters):
            for j, group_j in enumerate(clusters):
                # Compute the average distance between all points in cluster i and cluster j
                distances = [
                    distance_matrix[point_i, point_j]
                    for point_i in group_i
                    for point_j in group_j
                ]
                new_matrix[i, j] = np.mean(distances)
        current_matrix = new_matrix
    return results

    
