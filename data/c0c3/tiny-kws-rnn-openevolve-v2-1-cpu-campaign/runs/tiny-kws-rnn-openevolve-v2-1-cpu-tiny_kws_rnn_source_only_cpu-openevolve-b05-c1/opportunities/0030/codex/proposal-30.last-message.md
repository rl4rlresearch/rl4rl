MECHANISM: Full-rank single-retention-gate recurrence

HYPOTHESIS: A 96-unit full-rank single-gate recurrent state will retain at least 85% validation accuracy on the verified 26-frame max-augmented schedule while reducing total dense inference MACs by about 9.2%, from 521.65M to 473.82M.

INTENDED_EDIT: Replace the 81-unit three-gate GRU with a wider 96-unit recurrent update using one learned retention gate and one full-rank normalized proposal, while preserving mean-max-terminal classification and the successful frame schedule.

EVIDENCE: The current 81-unit GRU reaches 86.99%, providing accuracy margin, while the 128-unit rank-32 recurrent bottleneck failed at 82.70%. This motivates challenging the assumption that three gates are necessary without repeating the failed low-rank assumption: both retained state and proposal remain full-rank, and the wider state compensates for removing the reset gate.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with mean and terminal-state readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 81, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(243, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 81, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 81, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 81, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((mean_output, peak, hidden[:, 0, :]), dim=-1)
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        base_steps = min(28, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 27:
            del indices[1:3]
        return indices
=======
class KeywordSingleGate(nn.Module):
    """A full-rank single-gate recurrence with multi-view temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 96
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
        with torch.no_grad():
            self.transition.bias[: self.hidden_size].fill_(1.0)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        peak = torch.zeros_like(hidden)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        update = self.transition(
            torch.cat((self.input_norm(frame), hidden), dim=-1)
        )
        retention_logits, proposal_logits = update.chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
        hidden = retention * hidden + (1.0 - retention) * proposal
        return (
            hidden,
            summary + hidden,
            torch.maximum(peak, hidden),
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, peak, hidden), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        base_steps = min(28, available_frames)
        indices = [
            i * (available_frames - 1) // (base_steps - 1)
            for i in range(base_steps)
        ]
        if len(indices) > 27:
            del indices[1:3]
        return indices
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordGRU()
=======
def build_model() -> nn.Module:
    return KeywordSingleGate()
>>>>>>> REPLACE