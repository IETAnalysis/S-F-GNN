import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from DTTmodel import EncryptedTrafficClassifier
from jsd_feature_selection import jsd_feature_selector
from torch_geometric.data import Batch
import random
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def custom_collate_fn(batch):
    valid_items = [item for item in batch if item and 'session_graph' in item and item['session_graph'] is not None]
    if not valid_items: return None
    all_graphs_in_batch = []
    labels = []
    for item in valid_items:
        graph = item['session_graph']
        if 'attr' in graph.node_types and hasattr(graph['attr'], 'names'):
            del graph['attr'].names
        all_graphs_in_batch.append(graph)
        labels.append(item['label'])
    return {
        'batched_graphs': Batch.from_data_list(all_graphs_in_batch),
        'label': torch.tensor(labels, dtype=torch.long)
    }


def load_all_data(data_folder):
    data_files = [os.path.join(root, file) for root, _, files in os.walk(data_folder) for file in files if file.endswith('.pt')]
    all_data = []
    for data_path in tqdm(data_files, desc="Loading fine-tune data"):
        try:
            dataset = torch.load(data_path, weights_only=False)
            all_data.extend(dataset)
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
    print(f"Total data loaded: {len(all_data)} samples from {len(data_files)} files")
    return all_data


def select_labeled_from_train(train_data, label_ratio=0.1):
    original_dataset = train_data.dataset
    train_indices = train_data.indices
    total_labeled_count = max(1, int(len(train_indices) * label_ratio))
    label_to_indices = {}
    for idx in train_indices:
        label = original_dataset[idx]['label']
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    sorted_labels = sorted(label_to_indices.keys())
    num_classes = len(sorted_labels)

    if num_classes == 0:
        print("警告：训练数据中未找到任何类别。")
        return Subset(original_dataset, [])

    base_samples_per_class = total_labeled_count // num_classes
    remainder = total_labeled_count % num_classes

    print(f"计划在 {len(train_indices)} 个训练样本中，选取 {total_labeled_count} 个有标签样本 ({label_ratio * 100:.2f}%).")
    print(f"分布在 {num_classes} 个类别中。每个类别基础样本数: {base_samples_per_class}, 剩余样本数: {remainder} 将分配给前几个类别。")

    final_labeled_indices = []
    for i, label in enumerate(sorted_labels):
        num_to_sample = base_samples_per_class
        if i < remainder:
            num_to_sample += 1
        available_indices = label_to_indices[label]
        actual_samples_to_take = min(num_to_sample, len(available_indices))
        final_labeled_indices.extend(random.sample(available_indices, actual_samples_to_take))

    print(f"最终选取的有标签样本总数: {len(final_labeled_indices)} 个。")
    return Subset(original_dataset, final_labeled_indices)


def calculate_class_weights(dataset: Subset):
    original_dataset, indices = dataset.dataset, dataset.indices
    if not indices: return None
    class_counts = np.bincount([original_dataset[i]['label'] for i in indices])
    num_classes = len(class_counts)
    if num_classes == 0: return None
    weights = 1. / (class_counts + 1e-6)
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float)


def finetune_model(model, labeled_loader, test_loader, device, save_path, feature_stats, epochs=50, class_weights=None):
    best_acc = 0.0
    print("\n--- Starting Single-Stage Fine-tuning ---")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=000, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = FocalLoss(alpha=class_weights, gamma=2.2)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(labeled_loader, desc=f'Finetuning Epoch {epoch + 1}/{epochs}')
        for batch in pbar:
            if batch is None: continue
            batched_graphs = batch['batched_graphs'].to(device)
            labels = batch['label'].to(device)
            criterion.to(device)
            for node_type, x in batched_graphs.x_dict.items():
                if node_type in feature_stats and x.size(0) > 0:
                    mean = feature_stats[node_type]['mean'].to(device)
                    std = feature_stats[node_type]['std'].to(device) + 1e-8
                    normalized_x = (x - mean) / std
                    batched_graphs.x_dict[node_type] = torch.clamp(normalized_x, -10.0, 10.0)
            outputs = model({'session_graph': batched_graphs}, pretrain=False)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        metrics = test_model(model, test_loader, device, feature_stats)
        test_acc = metrics['accuracy']
        scheduler.step(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'best_acc': best_acc}, save_path)
            print(f"New best model saved (Epoch {epoch + 1}) - Test Acc: {test_acc:.4f}")

    print(f"Best Test Accuracy during training: {best_acc:.4f}")
    return model


def test_model(model, test_loader, device, feature_stats):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            if batch is None: continue
            batched_graphs = batch['batched_graphs'].to(device)
            labels = batch['label']
            for node_type, x in batched_graphs.x_dict.items():
                if node_type in feature_stats and x.size(0) > 0:
                    mean = feature_stats[node_type]['mean'].to(device)
                    std = feature_stats[node_type]['std'].to(device) + 1e-8
                    normalized_x = (x - mean) / std
                    batched_graphs.x_dict[node_type] = torch.clamp(normalized_x, -10.0, 10.0)
            outputs = model({'session_graph': batched_graphs}, pretrain=False)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        print("Test set is empty, skipping metrics calculation.")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).mean()
    precision = 100 * precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = 100 * recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = 100 * f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    class_report = classification_report(all_labels, all_predictions, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    np.set_printoptions(linewidth=1000)

    print(f'\n--- Test Results ---')
    print(f'Test Accuracy: {accuracy:.4f}%')
    print(f'Weighted Precision: {precision:.4f}%')
    print(f'Weighted Recall: {recall:.4f}%')
    print(f'Weighted F1-score: {f1:.4f}%')
    print('\nClassification Report:')
    print(class_report)
    print('\nConfusion Matrix:')
    print(conf_matrix)
    print('--- End Test Results ---\n')

    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'class_report': class_report, 'conf_matrix': conf_matrix
    }


def main():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    pretrained_path = ""
    stats_path = ""
    data_folder_a = ""
    finetuned_save_path = ""
    label_ratio = 000

    num_finetune_classes = 000

    feature_stats_full = torch.load(stats_path)
    checkpoint = torch.load(pretrained_path, map_location=device)
    config = checkpoint['config']
    all_data = load_all_data(data_folder_a)

    print("\n--- Running JSD feature selection on the current fine-tuning dataset ---")
    original_feature_dims = {'flow': 84, 'time': 6, 'attr': 16}
    stable_feature_map = {}
    for node_type, dim in original_feature_dims.items():
        stable_indices, unstable_indices = jsd_feature_selector(
            dataset=all_data,
            node_type=node_type,
            num_features=dim
        )
        stable_feature_map[node_type] = stable_indices
        print(f"  - Node '{node_type}': Selected {len(stable_indices)} out of {dim} features as stable.")

        if unstable_indices:
            print(f"  - Unstable feature indices for '{node_type}' to be excluded: {unstable_indices}")
        else:
            print(f"  - No unstable features found for '{node_type}'.")

    print("--- Feature selection complete ---\n")

    model = EncryptedTrafficClassifier(
        hidden_channels=config['hidden_channels'],
        out_channels=num_finetune_classes,
        node_types=config['node_types'],
        edge_types=config['edge_types'],
        num_clusters=config['num_clusters'],
        num_sage_layers=config.get('num_sage_layers', 000),
        num_gat_layers=config.get('num_gat_layers', 000)
    ).to(device)

    if all_data:
        print("Initializing model with a dummy forward pass...")
        dummy_sample = all_data[0]['session_graph'].clone()
        dummy_batch = Batch.from_data_list([dummy_sample]).to(device)
        model.eval()
        with torch.no_grad():
            model({'session_graph': dummy_batch})
        model.train()
        print("Model initialized successfully.")
    else:
        raise ValueError("Cannot initialize model, dataset is empty.")

    print("Adapting model's input layers and loading pretrained weights...")
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()

    for node_type, stable_indices in stable_feature_map.items():
        if node_type in model.static_model.encoder.in_lins:
            new_in_channels = len(stable_indices)
            if new_in_channels > 0:
                old_linear_layer = model.static_model.encoder.in_lins[node_type]
                new_linear_layer = nn.Linear(new_in_channels, old_linear_layer.out_channels).to(device)

                weight_key = f'static_model.encoder.in_lins.{node_type}.weight'
                bias_key = f'static_model.encoder.in_lins.{node_type}.bias'

                if weight_key in pretrained_dict:
                    stable_indices_tensor = torch.tensor(stable_indices, dtype=torch.long, device=device)
                    adapted_weights = pretrained_dict[weight_key].index_select(1, stable_indices_tensor)
                    new_linear_layer.weight.data.copy_(adapted_weights)

                if bias_key in pretrained_dict:
                    new_linear_layer.bias.data.copy_(pretrained_dict[bias_key])

                model.static_model.encoder.in_lins[node_type] = new_linear_layer

    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and 'in_lins' not in k and 'classifier' not in k and 'graph_projector' not in k
    }

    model.load_state_dict(pretrained_dict_filtered, strict=False)
    print("Pretrained static model adapted and loaded successfully.")
    print("Filtering fine-tuning dataset to use only stable features...")
    filtered_data = []
    for sample in tqdm(all_data, desc="Applying stable feature masks"):
        graph = sample.get('session_graph')
        if graph:
            new_graph = graph.clone()
            for node_type, stable_indices in stable_feature_map.items():
                if node_type in new_graph.node_types and hasattr(new_graph[node_type], 'x') and new_graph[node_type].x.size(0) > 0:
                    stable_indices_tensor = torch.tensor(stable_indices, dtype=torch.long)
                    if new_graph[node_type].x.size(1) == original_feature_dims[node_type]:
                        new_graph[node_type].x = new_graph[node_type].x.index_select(1, stable_indices_tensor)
                    else:
                        print(f"Warning: Mismatch in feature dimensions for node type {node_type}. Skipping filtering.")
            sample['session_graph'] = new_graph
        filtered_data.append(sample)

    feature_stats_stable = {}
    for node_type, stats in feature_stats_full.items():
        if node_type in stable_feature_map:
            stable_indices = torch.tensor(stable_feature_map[node_type], dtype=torch.long)
            feature_stats_stable[node_type] = {
                'mean': stats['mean'].index_select(0, stable_indices),
                'std': stats['std'].index_select(0, stable_indices)
            }

    train_size = int(len(filtered_data) * 0.9)
    total_train_data, total_test_data = random_split(filtered_data, [train_size, len(filtered_data) - train_size])

    labeled_train_data = select_labeled_from_train(total_train_data, label_ratio)
    class_weights = calculate_class_weights(labeled_train_data)
    labeled_loader = DataLoader(labeled_train_data, batch_size=000, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
    test_loader = DataLoader(total_test_data, batch_size=000, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

    print("Starting fine-tuning...")
    model = finetune_model(model, labeled_loader, test_loader, device, finetuned_save_path, feature_stats_stable, epochs=50, class_weights=class_weights)

    print("\n--- Testing Final Best Model ---")
    best_checkpoint = torch.load(finetuned_save_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    metrics = test_model(model, test_loader, device, feature_stats_stable)
    print(f"\n微调完成。最佳模型 - accuracy: {metrics['accuracy']:.4f}")
    print(f"微调完成。最佳模型 - precision: {metrics['precision']:.4f}")
    print(f"微调完成。最佳模型 - recall: {metrics['recall']:.4f}")
    print(f"微调完成。最佳模型 - f1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()