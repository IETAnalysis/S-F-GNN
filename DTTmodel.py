import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, GATv2Conv, SAGEConv, AttentionalAggregation


def calculate_node_importance(x_dict, edge_index_dict, node_types, alpha=0.5):
    node_importance = {}
    for node_type in node_types:
        device = x_dict[node_type].device
        num_nodes = x_dict[node_type].size(0)

        if num_nodes <= 1:
            node_importance[node_type] = torch.zeros(num_nodes, device=device)
            continue
        degree = torch.zeros(num_nodes, device=device)
        for edge_type, edge_idx in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if src_type == node_type:
                degree.scatter_add_(0, edge_idx[0], torch.ones_like(edge_idx[0], dtype=torch.float))
            if dst_type == node_type:
                degree.scatter_add_(0, edge_idx[1], torch.ones_like(edge_idx[1], dtype=torch.float))

        max_deg, min_deg = degree.max(), degree.min()
        degree_norm = (degree - min_deg) / (max_deg - min_deg) if max_deg.item() > min_deg.item() else torch.zeros_like(degree)

        feature_var = torch.var(x_dict[node_type], dim=1)
        max_var, min_var = feature_var.max(), feature_var.min()
        var_norm = (feature_var - min_var) / (max_var - min_var) if max_var.item() > min_var.item() else torch.zeros_like(feature_var)

        node_importance[node_type] = alpha * degree_norm + (1 - alpha) * var_norm
    return node_importance


class HeteroGraphSATEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_sage_layers, num_gat_layers, node_types, edge_types):
        super().__init__()

        self.num_sage_layers = num_sage_layers
        self.num_gat_layers = num_gat_layers
        self.in_lins = nn.ModuleDict({
            node_type: Linear(-1, hidden_channels) for node_type in node_types
        })
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_sage_layers):
            conv_dict = {
                edge_type: SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
                for edge_type in edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types}))

        for _ in range(num_gat_layers):
            conv_dict = {}
            edge_dims = {
                ('flow', 'acts_in', 'time'): 2, ('time', 'evolves_to', 'time'): 2,
                ('flow', 'interacts', 'flow'): 5, ('flow', 'uses', 'attr'): 1
            }
            for edge_type in edge_types:
                edge_dim = edge_dims.get(edge_type)
                conv_dict[edge_type] = GATv2Conv(
                    in_channels=hidden_channels, out_channels=hidden_channels, heads=4,
                    concat=False, add_self_loops=False, edge_dim=edge_dim
                )

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types}))

        self.out_lins = nn.ModuleDict({
            node_type: Linear(hidden_channels, out_channels) for node_type in node_types
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {
            key: self.in_lins[key](x)
            for key, x in x_dict.items() if x.size(0) > 0
        }

        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict_in = x_dict

            is_gat_layer = False
            first_key = next(iter(conv.convs.keys()), None)
            if first_key and isinstance(conv.convs[first_key], GATv2Conv):
                is_gat_layer = True

            if is_gat_layer:
                x_dict_out = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                x_dict_out = conv(x_dict, edge_index_dict)

            x_dict = {
                key: norm_dict[key](x_dict_in[key] + x_out) if key in x_dict_in and key in x_dict_out else x_out
                for key, x_out in x_dict_out.items()
            }
            x_dict = {key: F.gelu(x) for key, x in x_dict.items()}

        x_dict = {key: self.out_lins[key](x) for key, x in x_dict.items()}
        return x_dict


class SessionGraphEncoder(nn.Module):
    def __init__(self, hidden_channels, node_types, edge_types, num_sage_layers, num_gat_layers):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.encoder = HeteroGraphSATEncoder(
            hidden_channels, hidden_channels, num_sage_layers, num_gat_layers, node_types, edge_types
        )

    def forward(self, graph):
        x_dict = self.encoder(
            graph.x_dict, graph.edge_index_dict, getattr(graph, 'edge_attr_dict', None)
        )
        return x_dict


class EdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_src, x_dst):
        x = torch.cat([x_src, x_dst], dim=-1)
        return self.mlp(x)


class MaskedAutoencoder(nn.Module):
    def __init__(self, hidden_channels, mask_ratio=0.4, node_importance_alpha=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.node_importance_alpha = node_importance_alpha
        self.encoder = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 2), nn.ReLU(), nn.Linear(hidden_channels * 2, hidden_channels))
        self.decoder = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 2), nn.ReLU(), nn.Linear(hidden_channels * 2, hidden_channels))

    def forward(self, x_dict, graph):
        mask_dict = {}
        device = next(self.parameters()).device
        node_importance = calculate_node_importance(
            x_dict=x_dict, edge_index_dict=graph.edge_index_dict,
            node_types=x_dict.keys(), alpha=self.node_importance_alpha)
        for node_type, features in x_dict.items():
            num_nodes, num_features = features.shape
            if num_nodes == 0:
                mask_dict[node_type] = torch.empty(0, dtype=torch.bool, device=device)
                continue
            if node_type == 'flow':
                if num_features == 0:
                    mask_dict[node_type] = torch.zeros_like(features, dtype=torch.bool, device=device)
                    continue
                mask = torch.rand_like(features) < self.mask_ratio
                all_masked_nodes = torch.all(mask, dim=1)
                if torch.any(all_masked_nodes):
                    unmask_indices = torch.randint(0, num_features, (all_masked_nodes.sum(),), device=device)
                    rows_to_fix = torch.where(all_masked_nodes)[0]
                    mask[rows_to_fix, unmask_indices] = False
            else:
                mask_probs = self.mask_ratio * (1 - node_importance[node_type])
                mask = torch.bernoulli(mask_probs).bool()
                if not mask.any().item() and num_nodes > 0:
                    mask[torch.argmin(node_importance[node_type])] = True
            mask_dict[node_type] = mask
        x_masked_dict = {k: v.clone() for k, v in x_dict.items()}
        for node_type, mask in mask_dict.items():
            if node_type in x_masked_dict and x_masked_dict[node_type].numel() > 0:
                if mask.dim() == 2:
                    x_masked_dict[node_type].masked_fill_(mask, 0)
                else:
                    x_masked_dict[node_type][mask] = 0
        encoded_dict = {k: self.encoder(v) for k, v in x_masked_dict.items()}
        decoded_dict = {k: self.decoder(v) for k, v in encoded_dict.items()}
        return decoded_dict, mask_dict


class ClusterHead(nn.Module):
    def __init__(self, in_channels, num_clusters, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, in_channels))
        self.projector = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU(), nn.Linear(in_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x):
        z = self.projector(x)
        similarity = F.cosine_similarity(z.unsqueeze(1), self.cluster_centers.unsqueeze(0), dim=2) / self.temperature
        return similarity, z


class ContrastiveLossWithViews(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        device = z1.device
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        return self.criterion(logits, labels)

class EncryptedTrafficClassifier(nn.Module):
    def __init__(self, hidden_channels, out_channels, node_types, edge_types,
                 num_sage_layers=000, num_gat_layers=000, num_clusters=000, mask_ratio=000):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.static_model = SessionGraphEncoder(
            hidden_channels, node_types, edge_types, num_sage_layers, num_gat_layers
        )

        self.masked_autoencoder = MaskedAutoencoder(hidden_channels, mask_ratio=mask_ratio)
        self.pooler = AttentionalAggregation(gate_nn=Linear(hidden_channels, 1))

        self.fusion_attention_behavior = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), nn.Tanh(), nn.Linear(hidden_channels // 2, 1)
        )

        self.fusion_attention_final = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), nn.Tanh(), nn.Linear(hidden_channels // 2, 1)
        )

        graph_repr_dim = hidden_channels
        self.graph_cluster_head = ClusterHead(graph_repr_dim, num_clusters)
        self.contrastive_loss = ContrastiveLossWithViews(temperature=0.1)
        edge_dims = {
            ('flow', 'acts_in', 'time'): 2, ('time', 'evolves_to', 'time'): 2,
            ('flow', 'interacts', 'flow'): 5, ('flow', 'uses', 'attr'): 1
        }
        self.edge_predictors = nn.ModuleDict({
            f"{st}_{et}_{dt}": EdgePredictor(hidden_channels, hidden_channels, dim)
            for (st, et, dt), dim in edge_dims.items()
        })
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_repr_dim, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)
        )
        self.classifier = nn.Sequential(
            nn.Linear(graph_repr_dim, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels // 2), nn.BatchNorm1d(hidden_channels // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def get_hierarchical_graph_reps(self, graph, x_dict):
        num_graphs = graph.num_graphs
        device = next(self.parameters()).device

        behavior_features_pooled = []
        for node_type in ['flow', 'time']:
            if node_type in x_dict and x_dict[node_type].numel() > 0:
                features = x_dict[node_type]
                batch_vector = graph[node_type].batch
                if batch_vector is None: continue

                pooled = self.pooler(features, batch_vector)

                if pooled.size(0) < num_graphs:
                    filled_pooled = torch.zeros(num_graphs, features.size(1), device=device)
                    unique_batches = torch.unique(batch_vector)
                    filled_pooled[unique_batches] = pooled
                    pooled = filled_pooled
                behavior_features_pooled.append(pooled)

        behavior_repr = None
        if behavior_features_pooled:
            stacked_behavior = torch.stack(behavior_features_pooled, dim=1)
            attention_scores_behavior = self.fusion_attention_behavior(stacked_behavior)
            attention_weights_behavior = F.softmax(attention_scores_behavior, dim=1)
            behavior_repr = torch.sum(attention_weights_behavior * stacked_behavior, dim=1)

        attr_repr = None
        if 'attr' in x_dict and x_dict['attr'].numel() > 0:
            attr_features = x_dict['attr']
            batch_vector = graph['attr'].batch
            if batch_vector is not None:
                pooled_attr = self.pooler(attr_features, batch_vector)
                if pooled_attr.size(0) < num_graphs:
                    filled_pooled = torch.zeros(num_graphs, attr_features.size(1), device=device)
                    unique_batches = torch.unique(batch_vector)
                    filled_pooled[unique_batches] = pooled_attr
                    pooled_attr = filled_pooled
                attr_repr = pooled_attr

        final_reps_to_fuse = []
        if behavior_repr is not None:
            final_reps_to_fuse.append(behavior_repr)
        if attr_repr is not None:
            final_reps_to_fuse.append(attr_repr)

        if not final_reps_to_fuse:
            return None

        if len(final_reps_to_fuse) == 1:
            return final_reps_to_fuse[0]

        stacked_final = torch.stack(final_reps_to_fuse, dim=1)
        attention_scores_final = self.fusion_attention_final(stacked_final)
        attention_weights_final = F.softmax(attention_scores_final, dim=1)
        graph_repr = torch.sum(attention_weights_final * stacked_final, dim=1)

        return graph_repr

    def forward(self, data, pretrain=False, data_view_2=None):
        if pretrain:
            return self.pretrain_forward(data, data_view_2)

        session_graph = data['session_graph']
        x_dict_encoded = self.static_model(session_graph)
        graph_repr = self.get_hierarchical_graph_reps(session_graph, x_dict_encoded)

        if graph_repr is None:
            return torch.empty(0, self.classifier[-1].out_features, device=next(self.parameters()).device)

        final_logits = self.classifier(graph_repr)
        return final_logits

    def pretrain_forward(self, data, data_view_2=None):
        losses = {}
        graph_view_1 = data.get('session_graph')
        if not (graph_view_1 and any(nt in graph_view_1.x_dict and graph_view_1.x_dict[nt].numel() > 0 for nt in self.node_types if nt in graph_view_1.node_types)):
            return losses
        try:

            x_dict_encoded_1 = self.static_model(graph_view_1)
            graph_repr_1 = self.get_hierarchical_graph_reps(graph_view_1, x_dict_encoded_1)

            decoded_dict, mask_dict = self.masked_autoencoder(x_dict_encoded_1, graph_view_1)
            mae_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            total_masked_node_types = 0
            for node_type in decoded_dict.keys():
                mask = mask_dict[node_type]
                if mask.sum().item() > 0:
                    loss_term = F.mse_loss(decoded_dict[node_type][mask], x_dict_encoded_1[node_type][mask])
                    if not torch.isnan(loss_term):
                        mae_loss += loss_term
                        total_masked_node_types += 1
            if total_masked_node_types > 0: losses['mae'] = mae_loss / total_masked_node_types

            edge_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            num_edge_types_processed = 0
            for edge_type in graph_view_1.edge_types:
                src_type, rel_type, dst_type = edge_type
                key = f"{src_type}_{rel_type}_{dst_type}"
                if key not in self.edge_predictors: continue
                edge_index = graph_view_1[edge_type].edge_index
                edge_attr = graph_view_1[edge_type].edge_attr
                if edge_index.numel() == 0 or edge_attr is None: continue
                num_edges = edge_index.size(1)
                sample_size = min(num_edges, 16)
                if sample_size == 0: continue
                sampled_indices = torch.randperm(num_edges)[:sample_size]
                src_nodes, dst_nodes = edge_index[0][sampled_indices], edge_index[1][sampled_indices]
                src_embeds = x_dict_encoded_1[src_type][src_nodes]
                dst_embeds = x_dict_encoded_1[dst_type][dst_nodes]
                pred_edge_attr = self.edge_predictors[key](src_embeds, dst_embeds)
                true_edge_attr = edge_attr[sampled_indices]
                loss_term = F.mse_loss(pred_edge_attr, true_edge_attr)
                if not torch.isnan(loss_term):
                    edge_loss += loss_term
                    num_edge_types_processed += 1
            if num_edge_types_processed > 0:
                losses['edge'] = edge_loss / num_edge_types_processed

            if graph_repr_1 is not None and graph_repr_1.size(0) > 1:
                similarity, _ = self.graph_cluster_head(graph_repr_1)
                target = torch.argmax(similarity.detach(), dim=1)
                cluster_loss = F.cross_entropy(similarity, target)
                if not torch.isnan(cluster_loss):
                    losses['cluster'] = cluster_loss

            if data_view_2 is not None and graph_repr_1 is not None:
                graph_view_2 = data_view_2.get('session_graph')

                x_dict_encoded_2 = self.static_model(graph_view_2)
                graph_repr_2 = self.get_hierarchical_graph_reps(graph_view_2, x_dict_encoded_2)

                if graph_repr_2 is not None and graph_repr_1.size(0) > 1:
                    proj_features_1 = self.graph_projector(graph_repr_1)
                    proj_features_2 = self.graph_projector(graph_repr_2)
                    contrastive_loss = self.contrastive_loss(proj_features_1, proj_features_2)
                    if not torch.isnan(contrastive_loss): losses['contrastive'] = contrastive_loss
        except Exception as e:
            print(f"Pre-training forward pass failed with exception: {e}")
            pass
        return losses