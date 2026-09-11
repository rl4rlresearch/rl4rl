MECHANISM: Shared-transform input-gated recurrent cell with endpoint-and-trajectory readout

HYPOTHESIS: A 160-unit recurrent cell using one dense hidden transform and an input-conditioned retention gate will reach at least 85% accuracy while cutting per-step recurrent MACs from 36,000 to 32,000 versus the 100-unit GRU.

INTENDED_EDIT: Replace the GRU with a wider custom gated recurrence that shares one learned hidden transform, preserves information through an input-driven retention path, and classifies from both the final state and its temporal mean.

EVIDENCE: Full temporal coverage is load-bearing: 24- and 28-frame schedules failed, while full-rate GRUs reached 85.03–87.24%. The prior designs assumed three dense GRU transforms were necessary; this patch instead spends fewer MACs on a wider state with one recurrent transform and a richer temporal readout.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

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

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))


def build_model() -> nn.Module:
    return KeywordGRU()
=======
class SharedGateRNN(nn.Module):
    """A wide causal recurrent cell with one dense hidden transform."""

    hidden_size = 160

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.candidate_input = nn.Linear(20, self.hidden_size)
        self.retain_input = nn.Linear(20, self.hidden_size)
        self.recurrent = nn.Linear(self.hidden_size, self.hidden_size)
        self.candidate_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(2 * self.hidden_size, 8)
        nn.init.constant_(self.retain_input.bias, 1.0)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        frame = self.input_norm(frame)
        candidate = torch.tanh(
            self.candidate_norm(
                self.candidate_input(frame) + self.recurrent(hidden)
            )
        )
        retain = torch.sigmoid(self.retain_input(frame))
        hidden = retain * hidden + (1.0 - retain) * candidate
        return hidden, summary + hidden, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for frame in frames.unbind(dim=1):
            state = self.recurrent_step(frame, state)
        return state

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        temporal_mean = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((hidden, temporal_mean), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))


def build_model() -> nn.Module:
    return SharedGateRNN()
>>>>>>> REPLACE