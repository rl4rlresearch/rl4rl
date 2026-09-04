MECHANISM: Shared-bottleneck low-rank GRU recurrence

HYPOTHESIS: A 128-dimensional GRU-like state with rank-32 learned recurrent mixing and dual-timescale readout will retain at least 85% accuracy while reducing total inference MACs below the current 906,045,280.

INTENDED_EDIT: Replace the dense 98-unit standard GRU with a custom GRU update whose recurrent gates share a 32-dimensional bottleneck, while retaining all 32 frames and concatenating mean and final states for classification.

EVIDENCE: Dropping frames failed even at 24 steps, showing full temporal coverage is load-bearing, while the 98-unit mean-plus-final readout reached 87.12%. This challenges the remaining assumption that every gate needs full-rank hidden-to-hidden mixing: a wider state preserves representation capacity while low-rank recurrent matrices structurally reduce MACs.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(196, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
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
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
class KeywordGRU(nn.Module):
    """A causal GRU-like recurrence with shared low-rank state mixing."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 128
        self.input_norm = nn.LayerNorm(20)
        self.input_to_gates = nn.Linear(20, 3 * self.hidden_size)
        self.state_down = nn.Linear(self.hidden_size, 32, bias=False)
        self.state_to_gates = nn.Linear(32, 3 * self.hidden_size, bias=False)
        self.classifier = nn.Linear(2 * self.hidden_size, 8)

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
        x_reset, x_update, x_candidate = self.input_to_gates(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        recurrent_context = torch.tanh(self.state_down(hidden))
        h_reset, h_update, h_candidate = self.state_to_gates(
            recurrent_context
        ).chunk(3, dim=1)

        reset = torch.sigmoid(x_reset + h_reset)
        update = torch.sigmoid(x_update + h_update)
        candidate = torch.tanh(x_candidate + reset * h_candidate)
        hidden = update * hidden + (1.0 - update) * candidate
        return hidden, summary + hidden, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
>>>>>>> REPLACE