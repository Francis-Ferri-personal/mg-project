import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class OcularStatefulLSTM(nn.Module):
    """
    Stateful LSTM model for Ocular Signal Classification.
    Instead of treating windows independently, it propagates the hidden and cell states
    chronologically across consecutive feature windows belonging to the same patient.
    """
    def __init__(self, input_size=24, hidden_size=64, num_layers=2, num_classes=2, dropout=0.5):
        super(OcularStatefulLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Standard LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, n_reps, features) -> e.g., (batch_size, n_windows, 24)
        batch_size, n_reps, features = x.shape
        device = x.device

        if mask is not None:
            seq_lengths = mask.sum(dim=1).long()
        else:
            seq_lengths = torch.full((batch_size,), n_reps, dtype=torch.long, device=device)

        packed = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=n_reps)

        last_idx = (seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_size)
        last_valid_output = output.gather(1, last_idx).squeeze(1)

        logits = self.fc(self.dropout(last_valid_output))
        return logits