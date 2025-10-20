import numpy as np
from collections import defaultdict
from tqdm import tqdm

def get_probability_dist(data: np.ndarray, bins: int = 50, weights: np.ndarray = None) -> np.ndarray:
    if data.ndim > 1:
        data = data.flatten()

    finite_mask = np.isfinite(data)
    data = data[finite_mask]
    if weights is not None:
        weights = weights[finite_mask]

    if len(data) == 0:
        return np.zeros(bins)

    min_val, max_val = np.min(data), np.max(data)
    if min_val == max_val:
        hist = np.zeros(bins)
        hist[0] = 1.0
        return hist

    hist, _ = np.histogram(data, bins=bins, range=(np.min(data), np.max(data)), density=True, weights=weights)
    hist = hist.astype(np.float64)
    if np.sum(hist) > 0:
        return hist / np.sum(hist)
    return hist


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def jsd_feature_selector(dataset: list, node_type: str, num_features: int, label_key: str = 'label', sample_weights: list = None):
    print(f"\n--- Running Weighted JSD Feature Selection for node_type: '{node_type}' ---")
    if sample_weights is None:
        sample_weights = [1.0] * len(dataset)

    class_features = defaultdict(list)
    class_weights = defaultdict(list)
    for i, sample in enumerate(tqdm(dataset, desc=f"Extracting '{node_type}' features and weights")):
        graph = sample.get('session_graph')
        if graph and node_type in graph.node_types:
            features = graph[node_type].x.numpy()
            if features.shape[0] > 0:
                label = sample[label_key]
                class_features[label].append(features)
                class_weights[label].extend([sample_weights[i]] * features.shape[0])

    if not class_features:
        print(f"No features found for node_type '{node_type}'. Skipping.")
        return list(range(num_features)), []

    for label, feats in class_features.items():
        class_features[label] = np.concatenate(feats, axis=0)
        class_weights[label] = np.array(class_weights[label])

    labels = list(class_features.keys())
    feature_jsd_diffs = np.zeros(num_features)

    for feature_idx in tqdm(range(num_features), desc="Calculating JSD for each feature"):
        total_diff = 0
        total_weight = 0

        for label in labels:
            current_class_features = class_features[label][:, feature_idx]
            current_class_weights = class_weights[label]

            p = np.random.permutation(len(current_class_features))
            current_class_features = current_class_features[p]
            current_class_weights = current_class_weights[p]
            # --- 结束修改 ---

            split_point = len(current_class_features) // 2
            if split_point == 0: continue

            t1_data, t2_data = current_class_features[:split_point], current_class_features[split_point:]
            t1_weights, t2_weights = current_class_weights[:split_point], current_class_weights[split_point:]

            # 拼接其他所有类别的特征和权重
            other_classes_data = np.concatenate([
                class_features[l][:, feature_idx] for l in labels if l != label
            ])
            other_classes_weights = np.concatenate([
                class_weights[l] for l in labels if l != label
            ])

            if len(t1_data) == 0 or len(t2_data) == 0 or len(other_classes_data) == 0:
                continue

            p_t1 = get_probability_dist(t1_data, weights=t1_weights)
            p_t2 = get_probability_dist(t2_data, weights=t2_weights)
            p_t3 = get_probability_dist(other_classes_data, weights=other_classes_weights)

            inter_class_jsd = js_divergence(p_t1, p_t2)
            extra_class_jsd = js_divergence(p_t1, p_t3)
            diff = extra_class_jsd - inter_class_jsd

            weight = np.sum(current_class_weights)
            total_diff += diff * weight
            total_weight += weight

        if total_weight > 0:
            feature_jsd_diffs[feature_idx] = total_diff / total_weight

    stable_indices = np.where(feature_jsd_diffs >= 0)[0].tolist()
    unstable_indices = np.where(feature_jsd_diffs < 0)[0].tolist()

    print(f"JSD selection for '{node_type}' complete.")
    print(f"  - Found {len(stable_indices)} stable features.")
    print(f"  - Found {len(unstable_indices)} unstable features.")

    if not stable_indices:
        print("Warning: No stable features found. Defaulting to using all features.")
        return list(range(num_features)), []

    return stable_indices, unstable_indices