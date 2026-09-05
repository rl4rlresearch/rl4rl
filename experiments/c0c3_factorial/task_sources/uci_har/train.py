"""Editable small sensor-sequence classifier and training procedure."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Frozen upstream architecture configuration; training hyperparameters vary.
# are tuned via HPO. These values are locked for the research paper.
MICRO_BI_CONV_LSTM_CONFIG = {
    "convFilters": 16,
    "convKernel": 5,
    "convPadding": 2,
    "numConvLayers": 2,
    "poolSize": 2,
    "poolStride": 2,
    "lstmHidden": 24,
    "lstmLayers": 1,
    "bidirectional": True,
    "lstmOutput": 48,
}


class MicroBiConvLSTM(nn.Module):
    """Ultra-lightweight convolutional BiLSTM for human activity recognition.

    Stages:
        I.   Conv1D Stem:    inChannels -> convFilters, k=5, BatchNorm, ReLU, MaxPool.
        II.  Conv1D Block:   convFilters -> convFilters, k=5, BatchNorm, ReLU, MaxPool.
        III. BiLSTM:         convFilters -> lstmHidden * 2 (bidirectional).
        IV.  Aggregation:    Last timestep or mean pooling.
        V.   Classifier:     Linear(lstmOutput, numClasses).

    Args:
        numClasses:   Number of activity classes.
        inChannels:   Number of input sensor channels.
        seqLen:       Sequence length in timesteps.
        dropout:      Dropout rate before the classifier head.
        aggregation:  Temporal aggregation method ('last' or 'mean').
        convFilters:  Override conv filter count (for ablation studies only).
        lstmHidden:   Override LSTM hidden size (for ablation studies only).
    """

    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
        aggregation: str = "last",
        convFilters: int | None = None,
        lstmHidden: int | None = None,
    ):
        super().__init__()

        self.convFilters = convFilters or MICRO_BI_CONV_LSTM_CONFIG["convFilters"]
        self.convKernel = MICRO_BI_CONV_LSTM_CONFIG["convKernel"]
        self.convPadding = MICRO_BI_CONV_LSTM_CONFIG["convPadding"]
        self.poolSize = MICRO_BI_CONV_LSTM_CONFIG["poolSize"]
        self.lstmHidden = lstmHidden or MICRO_BI_CONV_LSTM_CONFIG["lstmHidden"]
        self.lstmLayers = MICRO_BI_CONV_LSTM_CONFIG["lstmLayers"]
        self.bidirectional = MICRO_BI_CONV_LSTM_CONFIG["bidirectional"]

        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.dropout = dropout
        self.aggregation = aggregation

        self.seqLenAfterPool = seqLen // 4
        self.lstmOutput = self.lstmHidden * 2 if self.bidirectional else self.lstmHidden

        # Stage I: Convolutional Stem.
        self.conv1 = nn.Conv1d(
            inChannels,
            self.convFilters,
            kernel_size=self.convKernel,
            padding=self.convPadding,
            stride=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(self.convFilters)
        self.pool1 = nn.MaxPool1d(kernel_size=self.poolSize, stride=self.poolSize)

        # Stage II: Second Convolutional Block.
        self.conv2 = nn.Conv1d(
            self.convFilters,
            self.convFilters,
            kernel_size=self.convKernel,
            padding=self.convPadding,
            stride=1,
            bias=True,
        )
        self.bn2 = nn.BatchNorm1d(self.convFilters)
        self.pool2 = nn.MaxPool1d(kernel_size=self.poolSize, stride=self.poolSize)

        # Stage III: Bidirectional LSTM.
        self.lstm = nn.LSTM(
            input_size=self.convFilters,
            hidden_size=self.lstmHidden,
            num_layers=self.lstmLayers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Stage V: Classification Head.
        self.dropoutLayer = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.lstmOutput, numClasses)

        self._initWeights()

    def _initWeights(self):
        """Initialize weights with best-practice defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [B, T, C]. Output: [B, numClasses]."""
        x = x.permute(0, 2, 1)  # [B, C, T]

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)  # [B, F, T/2]

        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool2(x)  # [B, F, T/4]

        x = x.permute(0, 2, 1)  # [B, T/4, F]
        x, _ = self.lstm(x)  # [B, T/4, lstmOutput]

        if self.aggregation == "last":  # noqa: SIM108
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)

        x = self.dropoutLayer(x)
        return self.classifier(x)


BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


def build_model():
    return MicroBiConvLSTM(numClasses=6, inChannels=9, seqLen=128, dropout=0.15)


def build_optimizer(model, total_steps):
    return torch.optim.AdamW(model.parameters(), lr=0.002185, weight_decay=0.000142)


def training_loss(model, features, labels, step, total_steps):
    return F.cross_entropy(model(features), labels)


def after_optimizer_step(optimizer, step, total_steps):
    import math

    rate = 1e-6 + (0.002185 - 1e-6) * 0.5 * (1 + math.cos(math.pi * step / total_steps))
    for group in optimizer.param_groups:
        group["lr"] = rate
