import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
  def __init__(self, input_size, hidden_size, seq_len, n_layers=2, dropout=0.5):
    super(LSTMPredictor, self).__init__()
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.n_layers = n_layers
    self.lstm = nn.LSTM(
      input_size=input_size,
      hidden_size=hidden_size,
      num_layers=n_layers,
      dropout=dropout,
      batch_first=True
    )
    self.linear = nn.Linear(in_features=hidden_size, out_features=1)

  def forward(self, sequences):
    # sequences: (batch_size, seq_len, input_size)
    batch_size = sequences.size(0)
    h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
    c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)

    lstm_out, _ = self.lstm(sequences, (h0, c0))
    # lstm_out: (batch_size, seq_len, hidden_size)
    last_time_step = lstm_out[:, -1, :]  # grab the last output from each sequence
    y_pred = self.linear(last_time_step)
    return y_pred
