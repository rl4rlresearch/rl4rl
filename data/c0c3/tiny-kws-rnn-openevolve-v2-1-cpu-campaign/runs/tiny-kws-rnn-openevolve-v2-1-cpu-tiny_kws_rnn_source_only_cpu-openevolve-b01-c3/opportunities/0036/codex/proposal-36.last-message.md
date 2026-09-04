MECHANISM: Full gate-feedback capacity with a narrower recurrent state

HYPOTHESIS: A 101-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 336,510,240 to approximately 333,211,120.

INTENDED_EDIT: Replace the dense 96-unit GRU with a 101-unit bottleneck-gated recurrence, preserving the 16-frame schedule and mean/final-state readout.

EVIDENCE: The adjacent 102-unit bottleneck-gated design qualified at 85.276% accuracy and 336,510,240 MACs; prior bottleneck-rank reductions were less reliable, so trimming state width by one is the closest supported structural reduction that preserves all demonstrated gate-feedback capacity.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
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
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
=======
class KeywordGRU(nn.Module):
    """A gated recurrent memory with low-dimensional feedback paths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 303)
        self.recurrent_down = nn.Linear(101, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 101) for _ in range(3)]
        )
        self.classifier = nn.Linear(202, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 101, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 101, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        hidden = hidden[:, 0, :]

        input_reset, input_update, input_candidate = self.input_projection(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.tanh(
            self.recurrent_down(hidden)
        ).chunk(3, dim=1)

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return hidden.unsqueeze(1), summary + hidden, count + 1.0

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
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE