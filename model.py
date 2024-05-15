import dgl
from dgl.nn.pytorch.gt import DegreeEncoder, PathEncoder, SpatialEncoder, BiasedMHA, GraphormerLayer
from dgl.batch import unbatch
from dgl import backend
import torch
from torch.nn.utils.rnn import pad_sequence
from dgl import shortest_dist
from torch import nn
from typing import Callable
from torch import Tensor
import math


@torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return nn.functional.dropout(nn.functional.softmax(input, -1), dropout_prob, is_training)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class SpatialEncoderBatch(nn.Module):
    def __init__(self, max_dist, num_heads=1):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        self.spatial_encoder = SpatialEncoder(max_dist=self.max_dist, num_heads=self.num_heads)

    def forward(self, bg):
        device = bg.device
        max_num_nodes = torch.max(bg.batch_num_nodes())
        g_list = unbatch(bg)
        n_graph = len(g_list)
        dist = -torch.ones((n_graph, max_num_nodes, max_num_nodes), dtype=torch.long).to(device)

        for i, ubg in enumerate(g_list):
            n_ubg = ubg.num_nodes()
            dist[i, :n_ubg, :n_ubg] = shortest_dist(ubg, root=None, return_paths=False)
        spatial_pos_bias = self.spatial_encoder(dist).to(device)
        return spatial_pos_bias


class PathEncoderBatch(nn.Module):
    def __init__(self, max_len, feat_dim, num_heads=1):
        super().__init__()
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.path_encoder = PathEncoder(self.max_len, self.feat_dim, self.num_heads)

    def forward(self, bg, edge_feat):
        g_list = unbatch(bg)
        n_graph = len(g_list)
        device = bg.device
        sum_num_edges = 0
        max_num_nodes = torch.max(bg.batch_num_nodes())
        path_encoding = torch.full((n_graph, max_num_nodes, max_num_nodes, self.num_heads),
                                   float(-1e3)).to(device)

        for i, ubg in enumerate(g_list[1:]):
            num_nodes = ubg.num_nodes()
            num_edges = ubg.num_edges()
            edata = edge_feat[sum_num_edges: (sum_num_edges + num_edges)]
            sum_num_edges = sum_num_edges + num_edges
            edata = torch.cat(
                (edata, torch.zeros(1, self.feat_dim).to(edata.device)), dim=0
            )
            dist, path = shortest_dist(ubg, root=None, return_paths=True)

            path_data = edata[path[:, :, :2]]
            out = self.path_encoder(dist.unsqueeze(0), path_data.unsqueeze(0))
            path_encoding[i, :num_nodes, :num_nodes] = out
        return path_encoding


class DegreeEncoderBatch(nn.Module):
    def __init__(self, max_degree, embedding_dim, direction="both"):
        super().__init__()
        self.direction = direction
        self.max_degree = max_degree
        self.direction = direction
        self.degree_encoder = DegreeEncoder(self.max_degree, embedding_dim, direction=self.direction)

    def forward(self, bg):
        g_list = unbatch(bg)
        device = bg.device
        if self.direction == 'in':
            in_degree = pad_sequence([g.in_degrees() for g in g_list], batch_first=True)
            degree_embedding = self.degree_encoder(in_degree).to(device)
        elif self.direction == 'out':
            out_degree = pad_sequence([g.out_degrees() for g in g_list], batch_first=True)
            degree_embedding = self.degree_encoder(out_degree).to(device)
        elif self.direction == "both":
            in_degree = pad_sequence([g.in_degrees() for g in g_list], batch_first=True)
            out_degree = pad_sequence([g.out_degrees() for g in g_list], batch_first=True)
            degree_embedding = self.degree_encoder(torch.stack((in_degree, out_degree))).to(device)
        else:
            raise ValueError(
                f'Supported direction options: "in", "out" and "both", '
                f"but got {self.direction}"
            )
        return degree_embedding


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs = softmax_dropout(
            attn + attn_bias, 0.1, self.training
        ).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)

        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]

        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)

        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)

        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        cur_force_gram = torch.bmm(cur_force, cur_force.transpose(2, 1))
        return cur_force_gram


class Graphormer(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        sp_num_heads,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        attention_dropout_rate,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.atom_encoder = nn.Embedding(1024*24+1, hidden_dim)
        self.spatial_pos_encoder = SpatialEncoderBatch(max_dist=2, num_heads=sp_num_heads)
        self.path_encoder = PathEncoderBatch(2, 7, num_heads=sp_num_heads)
        self.de_en = DegreeEncoderBatch(5, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [GraphormerLayer(ffn_dim, hidden_dim, num_heads, dropout=dropout_rate, attn_dropout=attention_dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        # self.graph_token = nn.Embedding(1, hidden_dim)
        # self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.hidden_dim = hidden_dim
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        self.node_proc = NodeTaskHead(hidden_dim, self.num_heads)

        self.down_stream_pj = nn.Linear(hidden_dim, 3)


    def forward(self, bg, x, attn_bias,attn_edge_type, perturb=None):
        degree_embedding = self.de_en(bg)

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(bg).permute([0,3,1,2])
        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        # reset spatial pos here
        # 所有节点都和虚拟节点直接有边相连，则所有节点和虚拟节点之间的最短路径长度为1
        # t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        # graph_attn_bias += t

        # edge feature
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        edge_input = self.path_encoder(bg, attn_edge_type).permute(0, 3, 1, 2)

        graph_attn_bias = graph_attn_bias + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x.long()).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        if perturb is not None:
            pass

        node_feature = node_feature + degree_embedding
        # graph_token_feature = self.graph_token.weight.unsqueeze(
        #     0).repeat(n_graph, 1, 1)
        # node_feature = torch.cat(
        #     [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias.permute(0, 2, 3, 1))
        output = self.final_ln(output)

        # delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        # dist = delta_pos.norm(dim=-1)
        # delta_pos /= dist.unsqueeze(-1) + 1e-5
        # node_output_gram1 = self.node_proc(output, graph_attn_bias, delta_pos)  # bsz_size n_node n_node
        output = self.down_stream_pj(output)
        k = output.view(-1, 3)
        node_output_gram = torch.bmm(output, output.permute([0, 2, 1]))
        # node_output_gram = node_output_gram1 + node_output_gram2
        return k, node_output_gram


    def get_feature(self, bg, x, attn_bias,attn_edge_type, perturb=None):
        degree_embedding = self.de_en(bg)

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(bg).permute([0,3,1,2])
        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        # reset spatial pos here
        # t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        # graph_attn_bias += t

        # edge feature
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        edge_input = self.path_encoder(bg, attn_edge_type).permute(0, 3, 1, 2)

        graph_attn_bias = graph_attn_bias + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x.long()).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        if perturb is not None:
            pass

        node_feature = node_feature + degree_embedding
        # graph_token_feature = self.graph_token.weight.unsqueeze(
        #     0).repeat(n_graph, 1, 1)
        # node_feature = torch.cat(
        #     [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias.permute(0, 2, 3, 1))
        features = self.final_ln(output)

        return features


class Graphormer_all(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        sp_num_heads,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        attention_dropout_rate,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.atom_encoder = nn.Embedding(1024*24+1, hidden_dim)
        self.spatial_pos_encoder = SpatialEncoderBatch(max_dist=2, num_heads=sp_num_heads)
        self.path_encoder = PathEncoderBatch(2, 7, num_heads=sp_num_heads)
        self.de_en = DegreeEncoderBatch(5, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [GraphormerLayer(ffn_dim, hidden_dim, num_heads, dropout=dropout_rate, attn_dropout=attention_dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.hidden_dim = hidden_dim
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        self.node_proc = NodeTaskHead(hidden_dim, self.num_heads)

        self.down_stream_pj = nn.Linear(hidden_dim, 3)


    def forward(self, bg, x, attn_bias,attn_edge_type, perturb=None):
        degree_embedding = self.de_en(bg)

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(bg).permute([0,3,1,2])
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        edge_input = self.path_encoder(bg, attn_edge_type).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x.long()).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        if perturb is not None:
            pass

        node_feature = node_feature + degree_embedding
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias.permute(0, 2, 3, 1))
        output = self.final_ln(output)

        # delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        # dist = delta_pos.norm(dim=-1)
        # delta_pos /= dist.unsqueeze(-1) + 1e-5
        # node_output_gram1 = self.node_proc(output, graph_attn_bias, delta_pos)  # bsz_size n_node n_node
        output = self.down_stream_pj(output)
        k = output[:, 1:, :].view(-1, 3)
        node_output_gram = torch.bmm(output[:, 1:, :], output[:, 1:, :].permute([0, 2, 1]))
        return k, node_output_gram


class Graphormer_finetune_regression(nn.Module):
    def __init__(
            self,
            n_layers,
            num_heads,
            hidden_dim,
            sp_num_heads,
            dropout_rate,
            intput_dropout_rate,
            ffn_dim,
            attention_dropout_rate,
            n_tasks=12,

    ):
        super().__init__()

        self.num_heads = num_heads
        self.atom_encoder = nn.Embedding(1024*24+1, hidden_dim)
        self.spatial_pos_encoder = SpatialEncoderBatch(max_dist=2, num_heads=sp_num_heads)
        self.path_encoder = PathEncoderBatch(2, 7, num_heads=sp_num_heads)
        self.de_en = DegreeEncoderBatch(5, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [GraphormerLayer(ffn_dim, hidden_dim, num_heads, dropout=dropout_rate, attn_dropout=attention_dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.pre_final_ln = nn.LayerNorm(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.pre_layers = nn.ModuleList([GraphormerLayer(ffn_dim, hidden_dim, num_heads, dropout=dropout_rate, attn_dropout=attention_dropout_rate)
                           for _ in range(1)])
        self.pre_graph_token = nn.Embedding(1, hidden_dim)
        self.pre_graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.hidden_dim = hidden_dim
        # self.node_proc = NodeTaskHead(hidden_dim, self.num_heads)

        # self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc1 = nn.Sequential(*[MLPBlock(hidden_dim, hidden_dim), Residual(MLPBlock(hidden_dim, hidden_dim))])
        # self.fc2 = nn.Linear(hidden_dim*4, hidden_dim)

        # self.fn = nn.Linear(hidden_dim, 64)
        # self.out_proj = nn.Linear(hidden_dim, n_tasks)
        self.pred1 = nn.Linear(hidden_dim, n_tasks)
        self.pred2 = nn.Linear(hidden_dim*2, n_tasks)

        self.apply(lambda module: init_params(module, n_layers=n_layers))


    def forward(self, bg, x, attn_bias,attn_edge_type, node_feature_3d=None, perturb=None):
        degree_embedding = self.de_en(bg)

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(bg).permute([0,3,1,2])
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        edge_input = self.path_encoder(bg, attn_edge_type).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x.long()).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        if perturb is not None:
            pass

        node_feature = node_feature + degree_embedding
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias.permute(0, 2, 3, 1))
        output = self.final_ln(output)
        if node_feature_3d is not None:
            pre_graph_token_feature = self.pre_graph_token.weight.unsqueeze(
                0).repeat(n_graph, 1, 1)
            node_feature_3d = torch.cat(
                [pre_graph_token_feature, node_feature_3d], dim=1)
            for pre_enc_layer in self.pre_layers:
                node_feature_3d = pre_enc_layer(node_feature_3d, attn_bias=None)
            node_feature_3d = self.pre_final_ln(node_feature_3d)

            output = torch.cat((output, node_feature_3d), 2)
            output = self.pred2(output[:, 0, :])
        else:
            output = self.pred1(output[:, 0, :])
        return output


class Graphormer_finetune(nn.Module):
    def __init__(
            self,
            n_layers,
            num_heads,
            hidden_dim,
            sp_num_heads,
            dropout_rate,
            intput_dropout_rate,
            ffn_dim,
            attention_dropout_rate,
            n_tasks=12,

    ):
        super().__init__()

        self.num_heads = num_heads
        self.atom_encoder = nn.Embedding(1024*24+1, hidden_dim)
        self.spatial_pos_encoder = SpatialEncoderBatch(max_dist=2, num_heads=sp_num_heads)
        self.path_encoder = PathEncoderBatch(2, 7, num_heads=sp_num_heads)
        self.de_en = DegreeEncoderBatch(5, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [GraphormerLayer(ffn_dim, hidden_dim, num_heads, dropout=dropout_rate, attn_dropout=attention_dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        # self.fc1 = nn.Sequential(*[MLPBlock(hidden_dim*2, 512), Residual(MLPBlock(512, 512)),
        #                          nn.Linear(512, n_tasks)])
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fn = nn.Linear(hidden_dim, 64)
        self.out_proj = nn.Linear(64, n_tasks)
        # self.out_proj = nn.Linear(hidden_dim, n_tasks)
        # self.pred1 = nn.Sequential(*[MLPBlock(hidden_dim, 512), Residual(MLPBlock(512, 512)),
        #                              nn.Linear(512, n_tasks)])

        self.apply(lambda module: init_params(module, n_layers=n_layers))


    def forward(self, bg, x, attn_bias,attn_edge_type, node_feature_3d=None, perturb=None):
        degree_embedding = self.de_en(bg)

        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(bg).permute([0,3,1,2])
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        edge_input = self.path_encoder(bg, attn_edge_type).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x.long()).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        if node_feature_3d is not None:
            node_feature = torch.cat((node_feature, node_feature_3d), 2)
            node_feature = self.fc1(node_feature)
            # node_feature = node_feature + node_feature_3d
        else:
            # node_feature = self.fc2(node_feature)
            pass
        if perturb is not None:
            pass

        node_feature = node_feature + degree_embedding
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias.permute(0, 2, 3, 1))
        output = self.final_ln(output)

        output = self.fn(output)
        output = self.out_proj(output[:, 0, :])

        # output = self.pred1(output[:, 0, :])

        # output = self.out_proj(output[:, 0, :])

        return output


