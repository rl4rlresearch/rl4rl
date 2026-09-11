MECHANISM: Nineteen-step endpoint-aware temporal readout

HYPOTHESIS: Reducing the qualified dual-readout design from 20 to 19 uniformly distributed frames will retain at least 85% validation accuracy while lowering total inference cost to approximately 882 million MACs and 19 recurrent steps per example.

INTENDED_EDIT: Add the proven zero-initialized terminal-state residual classifier and use 19 uniformly distributed causal frames.

EVIDENCE: The 20-step dual-readout model achieved 85.89% accuracy at 928,030,720 MACs, outperforming the 20-step mean-only model by 1.23 percentage points; its 0.89-point qualification margin motivates testing one fewer recurrent step.

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
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
        self.endpoint_classifier = nn.Linear(128, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(19, available_frames)
>>>>>>> REPLACE