MECHANISM: Temporal-bin deviation readout

HYPOTHESIS: A 68-unit, 23-step GRU augmented with four temporal-bin deviations will achieve at least 85% validation accuracy while using fewer dense inference MACs than the verified 67-unit, 24-step model.

INTENDED_EDIT: Use 23 frames and summarize recurrent outputs with final, maximum, global deviation, four temporal-bin means, and four temporal-bin deviations.

EVIDENCE: The 68-unit, 23-step four-bin-mean model reached 84.66% accuracy at 339,613,760 MACs; adding within-bin deviations supplies localized activation-duration information while keeping the estimated total near 341.4M MACs, below the qualifying 343,793,080-MAC design.

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
    """A compact causal GRU with global and temporal-bin statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 68, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(748, 8)
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
    ) -> tuple[torch.Tensor, ...]:
        hidden = torch.zeros(batch_size, 1, 68, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        square_summary = torch.zeros(batch_size, 68, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 68), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        bin_summary = torch.zeros(batch_size, 4, 68, device=device, dtype=dtype)
        bin_square_summary = torch.zeros(
            batch_size, 4, 68, device=device, dtype=dtype
        )
        bin_count = torch.zeros(batch_size, 4, 1, device=device, dtype=dtype)
        return (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summary,
            bin_square_summary,
            bin_count,
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
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summary,
            bin_square_summary,
            bin_count,
        ) = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        bin_index = torch.div(
            count * 4.0, 23.0, rounding_mode="floor"
        ).to(torch.long).clamp_max(3)
        bin_weight = F.one_hot(
            bin_index[:, 0], num_classes=4
        ).to(output.dtype).unsqueeze(-1)
        return (
            hidden.transpose(0, 1),
            summary + output,
            square_summary + output.square(),
            torch.maximum(running_max, output),
            count + 1.0,
            bin_summary + bin_weight * output.unsqueeze(1),
            bin_square_summary + bin_weight * output.square().unsqueeze(1),
            bin_count + bin_weight,
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
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summary,
            bin_square_summary,
            bin_count,
        ) = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        positions = count + torch.arange(
            frames.shape[1], device=frames.device, dtype=count.dtype
        ).unsqueeze(0)
        bin_index = torch.div(
            positions * 4.0, 23.0, rounding_mode="floor"
        ).to(torch.long).clamp_max(3)
        bin_weight = F.one_hot(
            bin_index, num_classes=4
        ).to(outputs.dtype)
        weighted_outputs = (
            outputs.unsqueeze(2) * bin_weight.unsqueeze(-1)
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            square_summary + outputs.square().sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            count + frames.shape[1],
            bin_summary + weighted_outputs.sum(dim=1),
            bin_square_summary + weighted_outputs.square().sum(dim=1),
            bin_count + bin_weight.sum(dim=1).unsqueeze(-1),
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
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            count,
            bin_summary,
            bin_square_summary,
            bin_count,
        ) = state
        safe_count = count.clamp_min(1.0)
        mean = summary / safe_count
        deviation = (
            square_summary / safe_count - mean.square()
        ).clamp_min(0.0).sqrt()
        safe_bin_count = bin_count.clamp_min(1.0)
        bin_mean = bin_summary / safe_bin_count
        bin_deviation = (
            bin_square_summary / safe_bin_count - bin_mean.square()
        ).clamp_min(0.0).sqrt()
        features = torch.cat(
            (
                hidden[:, 0, :],
                running_max,
                deviation,
                bin_mean.flatten(1),
                bin_deviation.flatten(1),
            ),
            dim=1,
        )
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        target_steps = min(24, available_frames)
=======
        target_steps = min(23, available_frames)
>>>>>>> REPLACE