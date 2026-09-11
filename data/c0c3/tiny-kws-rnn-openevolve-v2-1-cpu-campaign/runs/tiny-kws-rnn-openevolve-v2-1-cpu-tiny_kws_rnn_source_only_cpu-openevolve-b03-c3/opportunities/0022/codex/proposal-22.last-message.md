MECHANISM: Balanced three-way block-diagonal GRU with temporal-pyramid readout

HYPOTHESIS: A 47/47/48-unit four-bin model will retain at least 85% validation accuracy while reducing total inference MACs from 760,857,920 to approximately 751,834,240.

INTENDED_EDIT: Replace the global mean with four ordered eight-frame summaries and reduce two recurrent branches to 47 units, yielding a balanced 142-unit aggregate state.

EVIDENCE: The 47/48/48 temporal-pyramid design achieved 86.63% accuracy at 760,857,920 MACs; its 1.63-point margin motivates the smallest further width reduction, while balanced branch sizes minimize recurrent MACs for the resulting aggregate width.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three asymmetric parallel GRUs with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 49, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(145, 8)
=======
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an ordered causal temporal-pyramid readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(4 * 142, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden_a = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 145, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 4, 142, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        hidden_a, hidden_b, hidden_c, summary, count = state
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
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            summary + output,
            count + 1.0,
        )
=======
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
        bin_index = (count.to(dtype=torch.long) // 8).clamp(max=3)
        assignment = F.one_hot(bin_index[:, 0], num_classes=4).to(output.dtype)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            bins + assignment.unsqueeze(-1) * output.unsqueeze(1),
            bin_counts + assignment,
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        hidden_a, hidden_b, hidden_c, summary, count = state
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
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
=======
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
        bin_indices = (positions // 8).clamp(max=3)
        assignments = F.one_hot(bin_indices, num_classes=4).to(outputs.dtype)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        _hidden_a, _hidden_b, _hidden_c, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, bins, bin_counts, _count = state
        pooled = bins / bin_counts.clamp_min(1.0).unsqueeze(-1)
        return self.classifier(pooled.flatten(start_dim=1))
>>>>>>> REPLACE