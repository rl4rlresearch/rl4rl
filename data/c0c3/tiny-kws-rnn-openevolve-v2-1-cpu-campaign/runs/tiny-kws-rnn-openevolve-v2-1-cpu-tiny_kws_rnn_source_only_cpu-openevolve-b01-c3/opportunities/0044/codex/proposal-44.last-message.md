MECHANISM: Max-pooled candidate-biased gated-state width trim

HYPOTHESIS: A 96-unit state with 31/32/33 reset-update-candidate feedback and mean-max-endpoint readout will retain at least 85% validation accuracy while reducing total inference MACs from 320,647,080 to approximately 317,341,440.

INTENDED_EDIT: Reduce the qualified 97-unit recurrent state and readout to 96 units while preserving its 16-frame schedule, candidate-biased feedback allocation, temporal maximum pooling, and training procedure.

EVIDENCE: The adjacent 97-unit mean-max-endpoint design achieved 86.503% accuracy at 320,647,080 MACs, leaving a 1.503-point accuracy margin and making a one-unit state reduction the closest supported cost improvement.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A gated recurrent memory with selectively compressed feedback."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 312)
        self.recurrent_down = nn.Linear(104, 95)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 104),
                nn.Linear(32, 104),
                nn.Linear(32, 104),
            ]
        )
        self.classifier = nn.Linear(208, 8)
=======
class KeywordGRU(nn.Module):
    """A gated recurrent memory with low-dimensional feedback paths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 288)
        self.recurrent_down = nn.Linear(96, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 96),
                nn.Linear(32, 96),
                nn.Linear(33, 96),
            ]
        )
        self.classifier = nn.Linear(288, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 96), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
>>>>>>> REPLACE

<<<<<<< SEARCH
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 32), dim=1
        )
=======
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return hidden.unsqueeze(1), summary + hidden, count + 1.0
=======
        return (
            hidden.unsqueeze(1),
            summary + hidden,
            torch.maximum(maximum, hidden),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        features = torch.cat(
            (
                summary / count.clamp_min(1.0),
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(features)
>>>>>>> REPLACE