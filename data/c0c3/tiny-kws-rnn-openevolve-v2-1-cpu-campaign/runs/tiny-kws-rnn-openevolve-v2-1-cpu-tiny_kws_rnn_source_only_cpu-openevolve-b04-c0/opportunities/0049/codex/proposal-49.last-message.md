MECHANISM: Coarse temporal-pyramid recurrent readout

HYPOTHESIS: Four contiguous temporal-bin means will capture nonlinear phoneme timing omitted by the trend statistic, raising the 68-unit, 23-step GRU from 84.29% to at least 85% while remaining below the verified 67-unit, 24-step model’s dense MAC cost.

INTENDED_EDIT: Use 68 hidden units over 23 frames and replace the global mean readout with four causal temporal-bin means, retaining final, maximum, and deviation summaries.

EVIDENCE: The 68-unit, 23-step model had lower cross-entropy than the qualifying 67-unit, 24-step model, and adding a linear trend improved both accuracy and cross-entropy but remained short of 85%; a four-bin summary provides richer timing information while preserving the one-step MAC saving.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean, final, max, and deviation readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(268, 8)
=======
class KeywordGRU(nn.Module):
    """A compact causal GRU with temporal-bin, final, max, and deviation readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(476, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 67), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, square_summary, running_max, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        bin_summaries = torch.zeros(
            batch_size, 4, 68, device=device, dtype=dtype
        )
        bin_counts = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        return (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summaries,
            bin_counts,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        hidden, summary, square_summary, running_max, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            square_summary + output.square(),
            torch.maximum(running_max, output),
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
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summaries,
            bin_counts,
        ) = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        bucket = torch.clamp((count * 4.0 / 23.0).long(), max=3)
        weights = F.one_hot(bucket[:, 0], num_classes=4).to(output.dtype)
        return (
            hidden.transpose(0, 1),
            summary + output,
            square_summary + output.square(),
            torch.maximum(running_max, output),
            count + 1.0,
            bin_summaries + weights.unsqueeze(-1) * output.unsqueeze(1),
            bin_counts + weights,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        hidden, summary, square_summary, running_max, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            square_summary + outputs.square().sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
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
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summaries,
            bin_counts,
        ) = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        positions = count + torch.arange(
            frames.shape[1], device=frames.device, dtype=count.dtype
        ).unsqueeze(0)
        buckets = torch.clamp((positions * 4.0 / 23.0).long(), max=3)
        weights = F.one_hot(buckets, num_classes=4).to(outputs.dtype)
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            square_summary + outputs.square().sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            count + frames.shape[1],
            bin_summaries
            + (weights.unsqueeze(-1) * outputs.unsqueeze(2)).sum(dim=1),
            bin_counts + weights.sum(dim=1),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, square_summary, running_max, count = state
        safe_count = count.clamp_min(1.0)
        mean = summary / safe_count
        deviation = (
            square_summary / safe_count - mean.square()
        ).clamp_min(0.0).sqrt()
        features = torch.cat(
            (
                mean,
                hidden[:, 0, :],
                running_max,
                deviation,
            ),
            dim=1,
        )
        return self.classifier(features)
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summaries,
            bin_counts,
        ) = state
        safe_count = count.clamp_min(1.0)
        mean = summary / safe_count
        deviation = (
            square_summary / safe_count - mean.square()
        ).clamp_min(0.0).sqrt()
        bin_means = bin_summaries / bin_counts.clamp_min(1.0).unsqueeze(-1)
        features = torch.cat(
            (
                bin_means.flatten(1),
                hidden[:, 0, :],
                running_max,
                deviation,
            ),
            dim=1,
        )
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(24, available_frames)
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(23, available_frames)
>>>>>>> REPLACE