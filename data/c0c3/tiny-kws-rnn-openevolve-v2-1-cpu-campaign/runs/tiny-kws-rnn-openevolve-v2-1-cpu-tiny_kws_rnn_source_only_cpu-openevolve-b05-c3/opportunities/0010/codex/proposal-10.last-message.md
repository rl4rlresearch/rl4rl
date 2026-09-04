MECHANISM: Residual endpoint-aware temporal readout

HYPOTHESIS: A 128-unit GRU processing 20 uniform frames with a learned terminal-state residual head will achieve at least 85% accuracy while keeping total inference MACs below 0.93 billion.

INTENDED_EDIT: Replace the assumption that uniformly averaging recurrent outputs is sufficient with a dual-timescale prediction: retain the proven mean summary and add a zero-initialized classifier head over the terminal hidden state, which preserves ordered whole-utterance context without adding recurrent MACs.

EVIDENCE: The 20-step mean-only model reached 84.66%, just three validation examples below qualification. Unlike repeating temporal or width boundary searches, the endpoint head directly addresses the shared equal-averaging bottleneck while preserving the original predictor at initialization and adding only 834,560 validation MACs.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
=======
class KeywordGRU(nn.Module):
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
        self.endpoint_classifier = nn.Linear(128, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)
>>>>>>> REPLACE

<<<<<<< SEARCH
        steps = min(24, available_frames)
=======
        steps = min(20, available_frames)
>>>>>>> REPLACE