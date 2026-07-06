"""
models/lstm_model.py
--------------------
Stacked LSTM regression model for Remaining Useful Life prediction.

Architecture
------------
Input  : (batch, window_size, n_features)
         ↓
LSTM   : stacked, optional bidirectional
         ↓
Dropout
         ↓
FC-1   : hidden_size  → 64   (ReLU)
         ↓
FC-2   : 64           → 1    (linear)
         ↓
Output : scalar RUL prediction per sample
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMModel(nn.Module):
    """Stacked LSTM for RUL regression.

    Parameters
    ----------
    input_size:
        Number of input features per time step.
    hidden_size:
        Number of LSTM hidden units.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied *between* LSTM layers and before the FC
        head.  Ignored (set to 0) when ``num_layers == 1``.
    bidirectional:
        Whether to use a bidirectional LSTM.  The FC head automatically
        receives the concatenated forward + backward representations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM requires dropout=0 when num_layers==1 to avoid a PyTorch warning
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # input shape: (batch, seq, feature)
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # FC head: map LSTM output → 1 (RUL)
        fc_in = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Initialise weights for faster convergence
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform init for LSTM weights, zeros for biases."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # Set forget-gate bias to 1 to combat vanishing gradients
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RUL predictions.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, window_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Scalar RUL predictions of shape ``(batch,)``.
        """
        # out: (batch, seq_len, num_directions * hidden_size)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        out, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # Concatenate the last forward and backward hidden states
            # h_n[-2]: last layer forward; h_n[-1]: last layer backward
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*hidden)
        else:
            h = h_n[-1]  # (batch, hidden_size)

        h = self.dropout(h)
        out = self.fc(h)          # (batch, 1)
        return out.squeeze(-1)   # (batch,)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_feature_extractor(self) -> nn.Module:
        """Return only the LSTM trunk (useful for transfer-learning experiments)."""
        return self.lstm


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    device: Optional[torch.device] = None,
) -> LSTMModel:
    """Instantiate and optionally move the model to a device.

    Parameters
    ----------
    input_size:
        Number of input features.
    hidden_size:
        LSTM hidden dimension.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability.
    bidirectional:
        Use bidirectional LSTM.
    device:
        Target device; defaults to CPU.

    Returns
    -------
    LSTMModel
        Initialised model.
    """
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    if device is not None:
        model = model.to(device)
    return model
