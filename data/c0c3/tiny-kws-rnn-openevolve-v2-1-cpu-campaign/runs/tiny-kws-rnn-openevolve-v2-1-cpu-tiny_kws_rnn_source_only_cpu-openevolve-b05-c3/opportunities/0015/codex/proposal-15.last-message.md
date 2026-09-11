MECHANISM: One-unit width reduction of the endpoint-aware dual-readout GRU

HYPOTHESIS: A 124-unit GRU using 20 uniformly distributed frames and complementary mean/endpoint readouts will retain at least 85% validation accuracy while reducing total inference MACs to approximately 874,775,360.

INTENDED_EDIT: Replace the current 128-unit, 24-frame mean-only model with a 124-unit, 20-frame GRU whose logits combine the temporal mean and terminal hidden state.

EVIDENCE: The adjacent 125-unit dual-readout design qualified at 85.40% with 887,942,500 MACs, while the 126-unit version reached 86.87%; testing 124 units is the most direct remaining width-boundary probe.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
class KeywordGRU(nn.Module):
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(124, 8)
        self.endpoint_classifier = nn.Linear(124, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
>>>>>>> REPLACE