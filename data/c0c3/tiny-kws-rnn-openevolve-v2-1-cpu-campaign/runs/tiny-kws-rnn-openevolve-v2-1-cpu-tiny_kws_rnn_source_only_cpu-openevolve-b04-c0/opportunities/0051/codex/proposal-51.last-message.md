MECHANISM: Localized temporal-deviation readout

HYPOTHESIS: Four temporal-bin means and deviations will raise the 68-unit, 23-step GRU from 84.66% to at least 85% accuracy while keeping total inference below 343,793,080 MACs.

INTENDED_EDIT: Use a 68-unit GRU over 23 frames and accumulate causal first and second moments in four contiguous temporal bins for an 11-summary classifier.

EVIDENCE: The 68-unit, 23-step four-bin-mean model reached 84.66%; localized deviations add complementary duration information with an estimated total cost of 341,387,200 MACs.

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
    """A causal GRU with global and coarse temporal deviation summaries."""

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
    ) -> tuple[
        torch.Tensor,
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
        bin_summary = torch.zeros(
            batch_size, 4, 68, device=device, dtype=dtype
        )
        bin_square_summary = torch.zeros(
            batch_size, 4, 68, device=device, dtype=dtype
        )
        bin_count = torch.zeros(batch_size, 4, 1, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            hidden,
            summary,
            square_summary,
            running_max,
            bin_summary,
            bin_square_summary,
            bin_count,
            count,
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
        torch.Tensor,
    ]:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            bin_summary,
            bin_square_summary,
            bin_count,
            count,
        ) = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        bin_index = ((count.to(torch.long) * 4) // 23).clamp_max(3)
        membership = F.one_hot(
            bin_index[:, 0], num_classes=4
        ).to(output.dtype).unsqueeze(-1)
        return (
            hidden.transpose(0, 1),
            summary + output,
            square_summary + output.square(),
            torch.maximum(running_max, output),
            bin_summary + membership * output.unsqueeze(1),
            bin_square_summary + membership * output.square().unsqueeze(1),
            bin_count + membership,
            count + 1.0,
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
        torch.Tensor,
    ]:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            bin_summary,
            bin_square_summary,
            bin_count,
            count,
        ) = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        positions = count[:, 0].to(torch.long).unsqueeze(1) + torch.arange(
            frames.shape[1], device=frames.device
        ).unsqueeze(0)
        bin_indices = ((positions * 4) // 23).clamp_max(3)
        membership = F.one_hot(
            bin_indices, num_classes=4
        ).to(outputs.dtype)
        weights = membership.unsqueeze(-1)
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            square_summary + outputs.square().sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            bin_summary + (weights * outputs.unsqueeze(2)).sum(dim=1),
            bin_square_summary
            + (weights * outputs.square().unsqueeze(2)).sum(dim=1),
            bin_count + membership.sum(dim=1).unsqueeze(-1),
            count + frames.shape[1],
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
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            hidden,
            summary,
            square_summary,
            running_max,
            bin_summary,
            bin_square_summary,
            bin_count,
            count,
        ) = state
        safe_count = count.clamp_min(1.0)
        mean = summary / safe_count
        deviation = (
            square_summary / safe_count - mean.square()
        ).clamp_min(0.0).sqrt()
        safe_bin_count = bin_count.clamp_min(1.0)
        bin_means = bin_summary / safe_bin_count
        bin_deviations = (
            bin_square_summary / safe_bin_count - bin_means.square()
        ).clamp_min(0.0).sqrt()
        features = torch.cat(
            (
                hidden[:, 0, :],
                running_max,
                deviation,
                bin_means.flatten(start_dim=1),
                bin_deviations.flatten(start_dim=1),
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