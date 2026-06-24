import json
import torch
from torch.utils.data import Dataset



class AccessionDataset(Dataset):
    """
    Dataset for Myasthenia Gravis (MG) binary classification.
    Maps 'Healthy control' to class 0, and any MG variant ('Probable/Definite MG') to class 1.
    """
    def __init__(self, file_paths, window_size=240, frequency_key='freq_1.0', feature_key="speed_horizontalAVG"):
        self.file_paths = file_paths
        self.window_size = window_size
        self.frequency_key = frequency_key
        self.feature_key = feature_key
        # Binary classification mapping
        self.label_map = {
            'Healthy control': 0,
            'Definite MG': 1
        }
        self.samples = self._index_files()
        print(self.samples)

    def _index_files(self):
        samples = []
        for path in self.file_paths:
            label = -1
            for label_text, label_id in self.label_map.items():
                if label_text in path:
                    label = label_id
                    break
            
            if label == -1:
                continue

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                visits = data.get('visits', {})
                for visit_id, visit_data in visits.items():
                    for freq_id, freq_data in visit_data.items():
                        if freq_id == self.frequency_key:
                            samples.append({
                                'path': path,
                                'visit_id': visit_id,
                                'freq_id': freq_id,
                                'label': label
                            })
            except Exception as e:
                print(f"Error indexing file {path}: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        path = sample_info['path']
        visit_id = sample_info['visit_id']
        freq_id = sample_info['freq_id']
        label = sample_info['label']

        # Flexible encoding fallback strategy during loading
        data = None
        for enc in ['utf-16-le', 'utf-16', 'utf-8']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    data = json.load(f)
                break
            except Exception:
                continue

        if data is None:
            return torch.zeros((1, 256, 1)), torch.tensor(label)

        try:
            session_data = data['visits'][visit_id][freq_id]
            feature_key = self.feature_key
            if feature_key not in session_data:
                # Raise error
                raise ValueError(f"Feature key '{feature_key}' not found in session data")
            whole_cycle = session_data[feature_key]
            
            # Convert the raw array to a PyTorch tensor
            full_signal = torch.tensor(whole_cycle, dtype=torch.float32) # Shape: (total_points,)
            
            window_size = self.window_size
            total_points = full_signal.size(0)
            
            # Calculate how many full windows we can extract
            n_reps = total_points // window_size
            
            if n_reps == 0:
                # Fallback if the signal is shorter than the window size
                padding = torch.zeros(window_size - total_points)
                padded_signal = torch.cat([full_signal, padding])
                return padded_signal.view(1, window_size, 1), torch.tensor(label)
            
            # Truncate any leftover remainder points at the very end of the signal
            valid_length = n_reps * window_size
            full_signal = full_signal[:valid_length]
            
            # Chunk/Reshape the long array into (n_reps, window_size, 1)
            # .view(n_reps, window_size) splits it into rows of window_size points
            sequence = full_signal.view(n_reps, window_size, 1)
            
            return sequence, torch.tensor(label)

        except Exception as e:
            print(f"Error windowing sample from {path}: {e}")
            return torch.zeros((1, 256, 1)), torch.tensor(label)


def ocular_collate_fn(batch, window_size=240):
    """
    Custom collate function to pack items with a variable number of repetitions (n_reps) 
    into a uniformly padded tensor batch.
    """
    sequences, labels = zip(*batch)
    max_n = max([s.size(0) for s in sequences])
    
    padded_sequences = []
    masks = []
    for s in sequences:
        n = s.size(0)
        padding = torch.zeros((max_n - n, window_size, 1), dtype=s.dtype)
        padded_s = torch.cat([s, padding], dim=0)
        padded_sequences.append(padded_s)
        
        mask = torch.cat([torch.ones(n), torch.zeros(max_n - n)])
        masks.append(mask)

    return torch.stack(padded_sequences), torch.tensor(labels), torch.stack(masks)