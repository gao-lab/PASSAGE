'''
core model of PASSAGE
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .layers import AttentionModule, GATConv, MLP

cudnn.deterministic = True
cudnn.benchmark = True
pdist = nn.PairwiseDistance(p=2)

class GATE(torch.nn.Module):
    def __init__(self, filter_size_1, filter_size_2, dropout_rate, number_of_labels):
        super(GATE, self).__init__()
        self.filter_1 = filter_size_1
        self.filter_2 = filter_size_2
        self.dropout = dropout_rate
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        self.conv1 = GATConv(self.number_labels, self.filter_1, heads=1, concat=False,
                             dropout=self.dropout, add_self_loops=True, bias=True)
        self.conv2 = GATConv(self.filter_1, self.filter_2, heads=1, concat=False,
                             dropout=self.dropout, add_self_loops=True, bias=True)
        self.conv3 = GATConv(self.filter_2, self.filter_1, heads=1, concat=False,
                             dropout=self.dropout, add_self_loops=True, bias=True)
        self.conv4 = GATConv(self.filter_1, self.number_labels, heads=1, concat=False,
                             dropout=self.dropout, add_self_loops=True, bias=True)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        h2 = F.normalize(h2, dim=1)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True, tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)


class PASSAGE(torch.nn.Module):
    """
    core architecture of PASSAGE
    """
    def __init__(self, GATE_hidden_size_1, GATE_hidden_size_2,
                 attention_pool_size, number_of_labels, dropout_rate):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(PASSAGE, self).__init__()
        self.GATE_hidden_size_1 = GATE_hidden_size_1
        self.GATE_hidden_size_2 = GATE_hidden_size_2
        self.attention_pool_size = attention_pool_size
        self.number_labels = number_of_labels
        self.dropout = dropout_rate
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.GATE = GATE(self.GATE_hidden_size_1, self.GATE_hidden_size_2, self.dropout, self.number_labels)
        self.MLP = MLP(self.GATE_hidden_size_2, self.attention_pool_size, self.dropout)
        self.attention = AttentionModule(self.attention_pool_size)
        self.freeze = False

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        if self.freeze == False:
            edge_index = data["edge_index"]
            features = data["features"]
            latent_feature, output = self.GATE(features, edge_index)
            loss = F.mse_loss(features, output)
    
            return loss
        else:
            edge_index_1 = data["edge_index_1"]
            edge_index_2 = data["edge_index_2"]
            edge_index_3 = data["edge_index_3"] 
            features_1 = data["features_1"]
            features_2 = data["features_2"]
            features_3 = data["features_3"]
            
            abstract_features_1, _ = self.GATE(features_1, edge_index_1)
            abstract_features_2, _ = self.GATE(features_2, edge_index_2)
            abstract_features_3, _ = self.GATE(features_3, edge_index_3)

            abstract_features_1 = self.MLP(abstract_features_1)
            abstract_features_2 = self.MLP(abstract_features_2)
            abstract_features_3 = self.MLP(abstract_features_3)

            _, pooled_features_1 = self.attention(abstract_features_1)
            _, pooled_features_2 = self.attention(abstract_features_2)
            _, pooled_features_3 = self.attention(abstract_features_3)

            score_1 = pdist(pooled_features_1.T, pooled_features_2.T)
            score_2 = pdist(pooled_features_1.T, pooled_features_3.T)
            
            return score_1, score_2



