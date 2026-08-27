import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

class AccessionDataset(Dataset):
    """
    Dataset for Myasthenia Gravis (MG) binary classification using extracted features.

    Each visit is represented by a feature sequence over time.
    Each window is split into two halves, and for each half we extract:
      - maximum value
      - minimum value
      - position index of the maximum
      - position index of the minimum

    Each channel contributes 27 features per axis and frequency. The loader
    keeps L and R separate, so two axes and three frequencies produce
    2 * 3 * 2 * 27 = 324 features per movement cycle.
    """
    DEFAULT_FREQUENCIES = ['0.5', '0.75', '1.0']
    FEATURE_KEYS = [
        'pos_min', 'pos_max', 'pos_mean', 'pos_std', 'pos_skewness',
        'pos_energy', 'low_state_amplitude', 'high_state_amplitude',
        'low_state_std', 'high_state_std', 'low_state_mean',
        'high_state_mean', 'symmetry_ratio', 'movement_gain_iou',
        'relative_offset', 'rise_mean_vel', 'rise_max_vel', 'rise_std_vel',
        'rise_mean_acc', 'rise_max_acc', 'rise_std_acc', 'fall_mean_vel',
        'fall_max_vel', 'fall_std_vel', 'fall_mean_acc', 'fall_max_acc',
        'fall_std_acc',
    ]
    FEATURES_PER_AXIS_FREQUENCY = len(FEATURE_KEYS) * 2
    FREQUENCY_WINDOW_SIZES = {
        'freq_0.5': 480,
        'freq_0.75': 360,
        'freq_1.0': 240,
    }

    def __init__(self, file_paths, window_size=240, frequency_key='freq_1.0', feature_key="speed_horizontalAVG"):
        self.file_paths = file_paths
        self.window_size = window_size
        self.feature_key = feature_key

        if frequency_key == 'all':
            self.frequency_keys = self.DEFAULT_FREQUENCIES.copy()
        elif isinstance(frequency_key, (list, tuple)):
            self.frequency_keys = list(frequency_key)
        else:
            self.frequency_keys = [frequency_key]

        if not self.frequency_keys:
            self.frequency_keys = self.DEFAULT_FREQUENCIES.copy()

        self.label_map = {
            'Healthy control': 0,
            'Definite MG': 1
        }
        self.samples = self._index_files()

    def _index_files(self):
        samples = []
        for path in self.file_paths:
            path_obj = Path(path)
            label = -1
            for label_text, label_id in self.label_map.items():
                if label_text in str(path_obj):
                    label = label_id
                    break

            if label == -1:
                continue

            try:
                with open(path_obj, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'axes' in data:
                    samples.append({
                        'path': str(path_obj),
                        'visit_id': data.get('visit_id', path_obj.stem),
                        'label': label
                    })
                    continue

                visits = data.get('visits', {})
                for visit_id, visit_data in visits.items():
                    if not isinstance(visit_data, dict):
                        continue

                    available_freqs = [
                        freq for freq in self.frequency_keys
                        if freq in visit_data or freq.removeprefix('freq_') in visit_data
                    ]
                    if available_freqs:
                        samples.append({
                            'path': str(path_obj),
                            'visit_id': visit_id,
                            'label': label
                        })
            except Exception as e:
                print(f"Error indexing file {path_obj}: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def _extract_window_features(self, series, window_size):
        if len(series) >= window_size:
            window = list(series[:window_size])
        else:
            window = list(series) + [0.0] * (window_size - len(series))

        half = len(window) // 2
        first_half = window[:half]
        second_half = window[half:]

        features = []
        for offset, segment in [(0, first_half), (half, second_half)]:
            if len(segment) == 0:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            max_val = float(max(segment))
            min_val = float(min(segment))
            max_idx = segment.index(max(segment)) + offset
            min_idx = segment.index(min(segment)) + offset

            features.extend([max_val, min_val, float(max_idx), float(min_idx)])

        return features

    def _extract_sequence_features(self, series, window_size):
        features = []
        if not isinstance(series, (list, tuple)) or len(series) == 0:
            return features

        for start in range(0, len(series), window_size):
            window = series[start:start + window_size]
            features.append(self._extract_window_features(window, window_size))

        return features

    def _axis_cycle_vector(self, cycle: dict):
        if not isinstance(cycle, dict):
            return [0.0] * self.FEATURES_PER_AXIS_FREQUENCY

        left = cycle.get('L', {})
        right = cycle.get('R', {})
        if not isinstance(left, dict):
            left = {}
        if not isinstance(right, dict):
            right = {}

        return [
            float(side.get(feature_key, 0.0) or 0.0)
            for side in (left, right)
            for feature_key in self.FEATURE_KEYS
        ]

    def _build_formatted_sequence(self, data):
        axes = data.get('axes', {})
        if not axes:
            return []

        per_freq_vectors = []
        for axis_name in ('horizontal', 'vertical'):
            axis_data = axes.get(axis_name, {})
            for freq_id in self.frequency_keys:
                freq_data = axis_data.get(freq_id, axis_data.get(freq_id.removeprefix('freq_'), {}))
                cycles = freq_data.get('cycles', []) if isinstance(freq_data, dict) else []
                if not cycles:
                    per_freq_vectors.append([[0.0] * self.FEATURES_PER_AXIS_FREQUENCY])
                    continue

                per_freq_vectors.append([self._axis_cycle_vector(cycle) for cycle in cycles])

        if not per_freq_vectors:
            return []

        max_cycle_count = max(len(v) for v in per_freq_vectors)
        combined = []
        for t in range(max_cycle_count):
            step = []
            for vectors in per_freq_vectors:
                if t < len(vectors):
                    vector = vectors[t]
                    if len(vector) < self.FEATURES_PER_AXIS_FREQUENCY:
                        vector = vector + [0.0] * (self.FEATURES_PER_AXIS_FREQUENCY - len(vector))
                    step.extend(vector[:self.FEATURES_PER_AXIS_FREQUENCY])
                else:
                    step.extend([0.0] * self.FEATURES_PER_AXIS_FREQUENCY)
            combined.append(step)

        return combined if combined else [[0.0] * (2 * len(self.frequency_keys) * self.FEATURES_PER_AXIS_FREQUENCY)]

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        path = sample_info['path']
        visit_id = sample_info['visit_id']
        label = sample_info['label']

        data = None
        for enc in ['utf-16-le', 'utf-16', 'utf-8']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    data = json.load(f)
                break
            except Exception:
                continue

        num_features = 2 * len(self.frequency_keys) * self.FEATURES_PER_AXIS_FREQUENCY
        if data is None:
            return torch.zeros((1, num_features), dtype=torch.float32), torch.tensor(label)

        try:
            if 'axes' in data:
                sequence = self._build_formatted_sequence(data)
                if not sequence:
                    sequence = [[0.0] * num_features]
                feature_sequence = torch.tensor(sequence, dtype=torch.float32)
                if feature_sequence.shape[1] < num_features:
                    pad = torch.zeros((feature_sequence.shape[0], num_features - feature_sequence.shape[1]), dtype=torch.float32)
                    feature_sequence = torch.cat([feature_sequence, pad], dim=1)
                return feature_sequence, torch.tensor(label)

            visit_data = data['visits'][visit_id]
            freq_sequences = []
            max_t = 0

            for freq_id in self.frequency_keys:
                freq_data = visit_data.get(freq_id, visit_data.get(freq_id.removeprefix('freq_')))
                series = None
                if isinstance(freq_data, dict):
                    series = freq_data.get(self.feature_key)

                window_size = self.FREQUENCY_WINDOW_SIZES.get(freq_id, self.window_size)
                seq_features = self._extract_sequence_features(series, window_size)
                freq_sequences.append(seq_features)
                max_t = max(max_t, len(seq_features))

            sequence = []
            for t in range(max_t):
                step_features = []
                for seq_features in freq_sequences:
                    if t < len(seq_features):
                        step_features.extend(seq_features[t])
                    else:
                        step_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                sequence.append(step_features)

            if len(sequence) == 0:
                sequence = [[0.0] * num_features]

            feature_sequence = torch.tensor(sequence, dtype=torch.float32)
            return feature_sequence, torch.tensor(label)

        except Exception as e:
            print(f"Error processing sample from {path}: {e}")
            return torch.zeros((1, num_features), dtype=torch.float32), torch.tensor(label)


def ocular_collate_fn(batch):
    """
    Custom collate function to pack items with a variable number of repetitions (n_reps)
    into a uniformly padded tensor batch.

    Each sample is expected to be shaped (n_reps, features), where
    the feature vector length is 324 when both axes and all frequencies are used.
    """
    sequences, labels = zip(*batch)
    max_n = max([s.size(0) for s in sequences])
    n_features = sequences[0].size(1)

    padded_sequences = []
    masks = []
    for s in sequences:
        n = s.size(0)
        if n < max_n:
            padding = torch.zeros((max_n - n, n_features), dtype=s.dtype)
            s = torch.cat([s, padding], dim=0)
        padded_sequences.append(s)

        mask = torch.cat([torch.ones(n), torch.zeros(max_n - n)])
        masks.append(mask)

    return torch.stack(padded_sequences), torch.tensor(labels), torch.stack(masks)
