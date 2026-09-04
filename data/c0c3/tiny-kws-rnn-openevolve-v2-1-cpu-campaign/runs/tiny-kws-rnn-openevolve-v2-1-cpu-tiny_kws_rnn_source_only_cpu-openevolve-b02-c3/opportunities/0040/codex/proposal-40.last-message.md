MECHANISM: Shared bottleneck-controlled gated recurrence

HYPOTHESIS: An 84-channel gated state whose recurrent control is mediated by a shared 28-dimensional nonlinear bottleneck will achieve at least 85% validation accuracy on the qualified 27-frame schedule while reducing predicted total inference MACs from 497,398,575 to approximately 429,928,800.

INTENDED_EDIT: Replace the full-rank 75-unit GRU with a wider custom gated state update using counted `nn.Linear` projections and a compact shared recurrent controller; retain causal spectral deltas and mean-plus-final prediction, and adopt the qualified 27-frame span-preserving schedule.

EVIDENCE: The full-rank 69-unit GRU qualified at 85.77% with 497,398,575 MACs, and widths 69–73 all qualified on the same 27-frame representation, while 26-frame thinning failed. This suggests retaining 27 observations but challenges the load-bearing assumption that every gate requires an independent full-rank hidden-to-hidden matrix.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU supplied with explicit spectral motion."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 75, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(150, 8)
=======
class KeywordGRU(nn.Module):
    """A wide gated state controlled through a compact recurrent bottleneck."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_gates = nn.Linear(40, 252)
        self.state_encoder = nn.Linear(84, 28, bias=False)
        self.state_gates = nn.Linear(28, 252)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 75, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 75, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        return hidden, summary, count, previous
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        return hidden, summary, count, previous
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
        output, hidden = self.gru(
            features.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            count + 1.0,
            normalized,
        )
=======
    def _update_hidden(
        self, input_drive: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        controller = torch.tanh(self.state_encoder(hidden))
        state_drive = self.state_gates(controller)
        input_reset, input_update, input_candidate = input_drive.chunk(3, dim=1)
        state_reset, state_update, state_candidate = state_drive.chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + state_reset)
        update = torch.sigmoid(input_update + state_update)
        candidate = torch.tanh(input_candidate + reset * state_candidate)
        return update * hidden + (1.0 - update) * candidate

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1)
        hidden = self._update_hidden(self.input_gates(features), hidden)
        return hidden, summary + hidden, count + 1.0, normalized
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        first_delta = (normalized[:, 0, :] - previous) * has_previous
        deltas = torch.cat(
            (
                first_delta.unsqueeze(1),
                normalized[:, 1:, :] - normalized[:, :-1, :],
            ),
            dim=1,
        )
        features = torch.cat((normalized, deltas), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
            normalized[:, -1, :],
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, previous = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        first_delta = (normalized[:, 0, :] - previous) * has_previous
        deltas = torch.cat(
            (
                first_delta.unsqueeze(1),
                normalized[:, 1:, :] - normalized[:, :-1, :],
            ),
            dim=1,
        )
        features = torch.cat((normalized, deltas), dim=2)
        input_drives = self.input_gates(features)
        for index in range(frames.shape[1]):
            hidden = self._update_hidden(input_drives[:, index, :], hidden)
            summary = summary + hidden
        return (
            hidden,
            summary,
            count + frames.shape[1],
            normalized[:, -1, :],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        start = max(available_frames - 28, 0)
        return list(range(start, available_frames))
=======
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden), dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
        start = max(available_frames - 28, 0)
        window_frames = available_frames - start
        if window_frames <= target_steps:
            return list(range(start, available_frames))
        return [
            start + round(index * (window_frames - 1) / (target_steps - 1))
            for index in range(target_steps)
        ]
>>>>>>> REPLACE