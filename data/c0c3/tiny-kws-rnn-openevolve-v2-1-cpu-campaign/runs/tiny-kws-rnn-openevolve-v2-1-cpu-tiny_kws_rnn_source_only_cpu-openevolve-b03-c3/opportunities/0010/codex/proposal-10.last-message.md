MECHANISM: Diagonal-plus-low-rank input-gated recurrent state

HYPOTHESIS: A 192-dimensional residual state with a rank-56 recurrent correction and input-conditioned retention will preserve at least 85% accuracy across all 32 frames while reducing total inference MACs below the qualified 98-unit GRU’s 905,406,320.

INTENDED_EDIT: Replace the dense three-gate GRU with a high-capacity diagonal-memory cell whose cross-state interaction is structurally low-rank, and classify from both its final state and temporal mean.

EVIDENCE: Full-frame GRUs reached a width boundary at 98 units (85.03%) versus 97 units (84.42%), while 24- and 16-frame schedules stayed below 83%. This challenges the load-bearing assumption that further savings require less state or fewer frames: the patch retains every frame and expands state capacity while replacing three dense recurrent transforms with a gated diagonal path plus one rank-56 correction.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)
=======
class KeywordGRU(nn.Module):
    """A causal diagonal-plus-low-rank gated recurrent model."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_candidate = nn.Linear(20, 192)
        self.retention = nn.Linear(20, 192)
        self.recurrent_down = nn.Linear(192, 56)
        self.recurrent_up = nn.Linear(56, 192)
        self.classifier = nn.Linear(384, 8)
        nn.init.constant_(self.retention.bias, 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 192, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 192, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        normalized = self.input_norm(frame)
        recurrent_code = torch.tanh(self.recurrent_down(hidden))
        proposal = torch.tanh(
            self.input_candidate(normalized) + self.recurrent_up(recurrent_code)
        )
        retain = torch.sigmoid(self.retention(normalized))
        hidden = retain * hidden + (1.0 - retain) * proposal
        return hidden, summary + hidden, count + 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        temporal_mean = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((hidden, temporal_mean), dim=-1))
>>>>>>> REPLACE