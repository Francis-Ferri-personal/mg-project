import torch
import torch.nn as nn

class OcularStatefulLSTM(nn.Module):
    """
    Stateful LSTM model for Ocular Signal Classification.
    Instead of treating windows independently, it propagates the hidden and cell states
    chronologically across consecutive 256-point windows belonging to the same patient.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2, dropout=0.5):
        super(OcularStatefulLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Standard LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, n_reps, seq_len, features) -> e.g., (batch_size, n_windows, 256, 1)
        batch_size, n_reps, seq_len, features = x.shape
        device = x.device
        
        # 1. Initialize hidden (h) and cell (c) states to zero at the beginning of the sequence
        # Shape: (num_layers, batch_size, hidden_size)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # Tensor to accumulate and hold the final valid state for each patient in the batch
        last_valid_states = torch.zeros(batch_size, self.hidden_size).to(device)
        
        # 2. Iterate chronologically through each consecutive window (t)
        for t in range(n_reps):
            # Extract window 't' for all patients in the batch
            # Shape: (batch_size, seq_len, features) -> (batch_size, 256, 1)
            window_t = x[:, t, :, :]
            
            # Pass the current window and the previous step's accumulated states into the LSTM
            # lstm_out shape: (batch_size, seq_len, hidden_size)
            lstm_out, (h, c) = self.lstm(window_t, (h, c))
            
            # Extract the features from the very last timestep of this window (index -1)
            current_state = lstm_out[:, -1, :] # Shape: (batch_size, hidden_size)
            
            # 3. Dynamic Masking / Padding Handling:
            # If a patient's recording has already ended (mask == 0 for this window), 
            # freeze their state at the last valid window to prevent corruption from padding zeros.
            if mask is not None:
                # mask[:, t] indicates if window 't' is real (1) or padded remainder (0)
                m = mask[:, t].unsqueeze(-1) # Shape: (batch_size, 1)
                last_valid_states = m * current_state + (1 - m) * last_valid_states
            else:
                last_valid_states = current_state

        # 4. Final classification based strictly on the accumulated terminal state of the full signal
        output = self.dropout(last_valid_states)
        logits = self.fc(output)
        return logits