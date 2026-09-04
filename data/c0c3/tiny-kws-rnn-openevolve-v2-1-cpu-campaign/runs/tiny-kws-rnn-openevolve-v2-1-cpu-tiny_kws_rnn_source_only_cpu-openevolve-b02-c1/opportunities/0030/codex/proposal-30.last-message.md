MECHANISM: Delta-augmented unified GRU with dual-timescale readout

HYPOTHESIS: A single cross-coupled 79-unit GRU receiving both normalized spectra and explicit frame-to-frame deltas will retain at least 85% accuracy while reducing total inference MACs from 671,511,100 to approximately 667,479,295.

INTENDED_EDIT: Replace the two isolated GRU branches with one 79-unit GRU over 40-dimensional spectrum-plus-delta features, preserve all 29 frames, and classify from concatenated mean and final recurrent outputs.

EVIDENCE: The 100+100 tanh RNN failed at 71.78%, showing learned gating is load-bearing, while reductions in hidden width or temporal evidence also missed 85%. This patch retains full GRU gating and the successful schedule but challenges the assumption that two independent memories of absolute spectra are preferable to a unified recurrent state with explicit local dynamics; its per-step recurrent cost is slightly lower (28,203 versus 28,383 MACs).

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Asymmetric parallel causal GRUs with a shared online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a, hidden_b, summary, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        output = torch.cat((output_a[:, 0, :], output_b[:, 0, :]), dim=1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            summary + output,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a, hidden_b, summary, count = state
        normalized = self.input_norm(frames)
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        outputs = torch.cat((outputs_a, outputs_b), dim=2)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, summary, count = state
        coordinates = self.classifier(summary / count.clamp_min(1.0))
        coordinate_sum = coordinates.sum(dim=1, keepdim=True)
        scale = 1.0 / math.sqrt(8.0)
        correction = (scale * scale) / (1.0 - scale)
        return torch.cat(
            (
                coordinates - correction * coordinate_sum,
                scale * coordinate_sum,
            ),
            dim=1,
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
=======
class KeywordGRU(nn.Module):
    """Unified gated memory over spectral levels and explicit local dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 79, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(158, 7)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 79, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, previous, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        features = torch.cat((normalized, delta), dim=1).unsqueeze(1)
        output, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        current = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            normalized,
            summary + current,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype).unsqueeze(1)
        first_delta = (
            normalized[:, :1, :] - previous.unsqueeze(1)
        ) * has_previous
        remaining_deltas = normalized[:, 1:, :] - normalized[:, :-1, :]
        deltas = torch.cat((first_delta, remaining_deltas), dim=1)
        features = torch.cat((normalized, deltas), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            normalized[:, -1, :],
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, _previous, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        readout = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        coordinates = self.classifier(readout)
        coordinate_sum = coordinates.sum(dim=1, keepdim=True)
        scale = 1.0 / math.sqrt(8.0)
        correction = (scale * scale) / (1.0 - scale)
        return torch.cat(
            (
                coordinates - correction * coordinate_sum,
                scale * coordinate_sum,
            ),
            dim=1,
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames))
>>>>>>> REPLACE