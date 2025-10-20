import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from DTTmodel import EncryptedTrafficClassifier
from torch_geometric.data import Batch

def augment_graph_batch_random(batch, noise_level=000, p_mask=000, random_feature_ratio=000):
    new_batch = batch.clone()
    x_augmented_dict = {node_type: new_batch[node_type].x.clone() for node_type in new_batch.node_types if new_batch[node_type].x.numel() > 0}

    for node_type, x in x_augmented_dict.items():
        num_features = x.size(1)
        num_to_augment = int(num_features * random_feature_ratio)
        if num_to_augment == 0:
            continue

        perm = torch.randperm(num_features, device=x.device)
        random_indices = perm[:num_to_augment]

        noise = torch.randn_like(x) * noise_level
        noise_mask = torch.zeros_like(x)
        noise_mask.index_add_(1, random_indices, noise.index_select(1, random_indices))
        x_augmented_dict[node_type] += noise_mask


        mask = torch.rand_like(x) < p_mask
        final_mask = torch.zeros_like(x, dtype=torch.bool)
        final_mask.index_add_(1, random_indices, mask.index_select(1, random_indices).type(torch.bool))
        x_augmented_dict[node_type][final_mask] = 0.0

    for node_type, x_aug in x_augmented_dict.items():
        new_batch[node_type].x = x_aug

    return new_batch


def custom_collate_fn(batch):
    valid_batch = [item for item in batch if item and 'session_graph' in item and item['session_graph'] is not None]
    if not valid_batch: return None
    session_graphs = [item['session_graph'] for item in valid_batch]
    for g in session_graphs:
        if 'attr' in g.node_types and hasattr(g['attr'], 'names'):
            del g['attr'].names
    labels = [item['label'] for item in valid_batch]
    return {
        'session_graph': Batch.from_data_list(session_graphs),
        'label': torch.tensor(labels, dtype=torch.long)
    }


def load_pretrain_data(data_folder):
    data_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(data_folder) for f in fn if f.endswith('.pt')]
    all_data = []
    for data_path in tqdm(data_files, desc="Loading all data for JSD"):
        try:
            dataset = torch.load(data_path, weights_only=False)
            all_data.extend(dataset)
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
    print(f"Total data loaded: {len(all_data)} samples from {len(data_files)} files")
    return all_data


def calculate_and_save_stats(dataset, stats_path):
    if os.path.exists(stats_path):
        print(f"Feature statistics file already exists at {stats_path}, skipping calculation.")
        return
    print("Calculating feature statistics on FULL feature set...")
    feature_sums, feature_sq_sums, feature_counts = defaultdict(float), defaultdict(float), defaultdict(int)
    for sample in tqdm(dataset, desc="Analyzing features"):
        graph = sample.get('session_graph')
        if graph:
            for node_type, x in graph.x_dict.items():
                if x is not None and x.numel() > 0:
                    feature_sums[node_type] += x.sum(dim=0).numpy()
                    feature_sq_sums[node_type] += (x ** 2).sum(dim=0).numpy()
                    feature_counts[node_type] += x.size(0)
    stats = {}
    for node_type in feature_counts:
        if feature_counts[node_type] == 0: continue
        mean = feature_sums[node_type] / feature_counts[node_type]
        variance = np.maximum(0, feature_sq_sums[node_type] / feature_counts[node_type] - mean ** 2)
        std = np.sqrt(variance)
        std[std < 1e-8] = 1.0
        stats[node_type] = {'mean': torch.tensor(mean, dtype=torch.float32), 'std': torch.tensor(std, dtype=torch.float32)}
    torch.save(stats, stats_path)
    print(f"Feature statistics saved to {stats_path}")


def is_batch_valid(batch):
    return 'session_graph' in batch and batch['session_graph'] is not None


def pretrain_model(model, dataloader, device, save_path, epochs, stats_path, hidden_channels_config):
    feature_stats = torch.load(stats_path, weights_only=False)
    for node_type in feature_stats:
        feature_stats[node_type]['mean'] = feature_stats[node_type]['mean'].to(device)
        feature_stats[node_type]['std'] = feature_stats[node_type]['std'].to(device)
    print("Feature statistics loaded and moved to device.")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=000, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    best_loss = float('inf')
    loss_weights = {'mae': 1.0, 'contrastive': 0.001, 'cluster': 1.0, 'edge': 0.1}

    for epoch in range(epochs):
        loss_accumulators = defaultdict(float)
        loss_counts = defaultdict(int)
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

        for batch in pbar:
            if not batch or not is_batch_valid(batch): continue
            original_graphs = batch['session_graph']

            # 使用新的随机增强函数
            batch_view_1 = augment_graph_batch_random(original_graphs, noise_level=000, p_mask=000, random_feature_ratio=000).to(device)
            batch_view_2 = augment_graph_batch_random(original_graphs, noise_level=000, p_mask=000, random_feature_ratio=000).to(device)

            for x in [batch_view_1, batch_view_2]:
                for node_type, data in x.x_dict.items():
                    if node_type in feature_stats and data.size(0) > 0:
                        mean = feature_stats[node_type]['mean'].to(device)
                        std = feature_stats[node_type]['std'].to(device) + 1e-8
                        normalized_x = (data - mean) / std
                        x.x_dict[node_type] = torch.clamp(normalized_x, -10.0, 10.0)

            loss_dict = model({'session_graph': batch_view_1}, pretrain=True, data_view_2={'session_graph': batch_view_2})

            if not loss_dict: continue
            total_loss_for_batch = torch.tensor(0.0, device=device)
            for name, loss_val in loss_dict.items():
                if name in loss_weights and loss_weights[name] > 0 and not torch.isnan(loss_val):
                    total_loss_for_batch += loss_weights[name] * loss_val
                    loss_accumulators[name] += loss_val.item()
                    loss_counts[name] += 1
            if total_loss_for_batch.item() == 0 or torch.isnan(total_loss_for_batch): continue

            optimizer.zero_grad()
            total_loss_for_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_accumulators['total'] += total_loss_for_batch.item()
            loss_counts['total'] += 1
            pbar.set_postfix({'batch_loss': f'{total_loss_for_batch.item():.6f}'})

        avg_total_loss = loss_accumulators['total'] / loss_counts['total'] if loss_counts['total'] > 0 else 0
        print(f"\n--- Epoch {epoch + 1}/{epochs} Summary ---")
        print(f"Average Total Weighted Loss: {avg_total_loss:.10f}")
        for name in ['mae', 'contrastive', 'cluster', 'edge']:
            if loss_counts[name] > 0:
                print(f"  - Avg {name} Loss: {loss_accumulators[name] / loss_counts[name]:.10f} (computed on {loss_counts[name]}/{len(dataloader)} batches)")
        print("--------------------------------------")

        scheduler.step(avg_total_loss)
        if 0 < avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'hidden_channels': hidden_channels_config,
                    'num_sage_layers': model.static_model.encoder.num_sage_layers,
                    'num_gat_layers': model.static_model.encoder.num_gat_layers,
                    'node_types': model.node_types, 'edge_types': model.edge_types,
                    'num_clusters': model.graph_cluster_head.cluster_centers.size(0)
                }
            }, save_path)
            print(f"New best generic model saved at epoch {epoch + 1} with loss {best_loss:.10f}")
    return model


def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    pretrain_data_folder = ""
    pretrain_save_path = ""
    stats_path = ""

    print("Loading pretraining data...")
    pretrain_dataset = load_pretrain_data(pretrain_data_folder)
    if not pretrain_dataset:
        print("Pretraining data could not be loaded. Exiting.")
        return

    print("\nCalculating stats on FULL features (using pre-training data)...")
    calculate_and_save_stats(pretrain_dataset, stats_path)

    dataloader = DataLoader(pretrain_dataset, batch_size=000, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)

    node_types = ['flow', 'time', 'attr']
    edge_types = [('flow', 'acts_in', 'time'), ('time', 'evolves_to', 'time'), ('flow', 'interacts', 'flow'), ('flow', 'uses', 'attr')]
    hidden_dim = 256
    NUM_PRETRAIN_CLASSES = 000

    model = EncryptedTrafficClassifier(
        hidden_channels=hidden_dim, out_channels=NUM_PRETRAIN_CLASSES,
        node_types=node_types, edge_types=edge_types,
        num_sage_layers=000, num_gat_layers=000,
        num_clusters=NUM_PRETRAIN_CLASSES, mask_ratio=000
    ).to(device)

    print(f"Model moved to device: {device}")
    print("Starting pretraining with a GENERIC framework...")
    pretrain_model(model, dataloader, device, pretrain_save_path, epochs=000,
                   stats_path=stats_path, hidden_channels_config=hidden_dim)
    print(f"Generic pretraining completed. Best model saved to {pretrain_save_path}")


if __name__ == "__main__":
    main()