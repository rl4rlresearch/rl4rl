MECHANISM: Phase-aware diagonal delta memory

HYPOTHESIS: A 128-channel input-conditioned diagonal memory using spectral deltas and explicit phase will retain at least 85% accuracy at roughly 213M dense MACs, despite removing the full-rank hidden-to-hidden transform.

INTENDED_EDIT: Replace the 83-unit full-rank recurrence with 128 independent learned-timescale memories driven by current mel features, frame differences, and temporal phase; retain the verified 24-step schedule and mean-max-terminal readout.

EVIDENCE: The current 24-step full-rank model reaches 86.13% at 336.06M MACs. The failed 128-unit rank-32 bottleneck reached 82.70%, suggesting hidden information compression was harmful; this patch instead preserves 128 direct memory channels and supplies explicit local dynamics while eliminating the costly recurrent matrix entirely.

<<<<<<< SEARCH
class KeywordSingleGate(nn.Module):
    """A full-rank single-gate recurrence with multi-view temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 83
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
        with torch.no_grad():
            self.transition.bias[: self.hidden_size].fill_(1.0)
=======
class KeywordDiagonalMemory(nn.Module):
    """Delta-aware diagonal memories with explicit temporal phase."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 128
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(42, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
        with torch.no_grad():
            self.transition.bias[: self.hidden_size].copy_(
                torch.linspace(-1.0, 3.0, self.hidden_size)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        peak = -torch.ones_like(hidden)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, previous, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden, summary, peak, previous, count = state
        normalized = self.input_norm(frame)
        phase = count / 23.0
        temporal_input = torch.cat(
            (normalized, normalized - previous, phase, phase * phase), dim=-1
        )
        retention_logits, proposal_logits = self.transition(
            temporal_input
        ).chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
        hidden = retention * hidden + (1.0 - retention) * proposal
        return (
            hidden,
            summary + hidden,
            torch.maximum(peak, hidden),
            normalized,
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, peak, hidden), dim=-1))
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        hidden, summary, peak, previous, count = state
        del previous
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, peak, hidden), dim=-1))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordSingleGate()
=======
def build_model() -> nn.Module:
    return KeywordDiagonalMemory()
>>>>>>> REPLACE