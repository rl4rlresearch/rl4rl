MECHANISM: Balanced three-way block-diagonal GRU with eight-bin causal temporal readout

HYPOTHESIS: Three 47-unit GRUs with eight ordered temporal bins will retain at least 85% validation accuracy while reducing total inference MACs from 755,537,600 to approximately 746,487,840.

INTENDED_EDIT: Replace the 128-unit GRU with three parallel 47-unit GRUs and classify eight four-frame temporal summaries.

EVIDENCE: The 47/47/48 eight-bin design achieved 86.13% accuracy, ten correct examples above the otherwise identical four-bin design and nine above qualification; removing one unit from its largest branch is the smallest recurrent-cost reduction and produces balanced branches.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
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
=======
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 141, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 141, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 8, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        output_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        output_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        output_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        output = torch.cat(
            (output_a[:, 0, :], output_b[:, 0, :], output_c[:, 0, :]),
            dim=-1,
        )
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=7)
        assignment = F.one_hot(bin_index[:, 0], num_classes=8).to(output.dtype)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + assignment.unsqueeze(-1) * output.unsqueeze(1),
            bin_counts + assignment,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        hidden_a, hidden_b, hidden_c, bins, bin_counts, count = state
        normalized = self.input_norm(frames)
        outputs_a, hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        outputs_b, hidden_b = self.gru_b(
            normalized, hidden_b.transpose(0, 1).contiguous()
        )
        outputs_c, hidden_c = self.gru_c(
            normalized, hidden_c.transpose(0, 1).contiguous()
        )
        outputs = torch.cat((outputs_a, outputs_b, outputs_c), dim=-1)
        positions = count.to(dtype=torch.long) + torch.arange(
            frames.shape[1], device=frames.device
        ).view(1, -1)
        bin_indices = (positions // 4).clamp(max=7)
        assignments = F.one_hot(bin_indices, num_classes=8).to(outputs.dtype)
        bin_sums = (
            outputs.unsqueeze(2) * assignments.unsqueeze(-1)
        ).sum(dim=1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + bin_sums,
            bin_counts + assignments.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, bins, bin_counts, _count = state
        pooled = bins / bin_counts.clamp_min(1.0).unsqueeze(-1)
        return self.classifier(pooled.flatten(start_dim=1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
>>>>>>> REPLACE