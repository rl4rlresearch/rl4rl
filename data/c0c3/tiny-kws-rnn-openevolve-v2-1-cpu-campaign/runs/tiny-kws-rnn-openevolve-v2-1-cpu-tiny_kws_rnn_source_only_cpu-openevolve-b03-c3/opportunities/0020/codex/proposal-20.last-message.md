MECHANISM: Four-bin causal temporal-pyramid readout

HYPOTHESIS: Three 48-unit GRUs with separate eight-frame temporal summaries will exceed 85% accuracy while using approximately 769,881,600 MACs, below the qualified 49/48/48 model.

INTENDED_EDIT: Replace the two 64-unit branches and global mean with three 48-unit branches whose outputs are pooled into four ordered causal segments before classification.

EVIDENCE: Three 48-unit GRUs missed qualification by one validation example at 767,064,960 MACs, while 49/48/48 qualified; this tests whether global temporal averaging—not recurrent capacity—caused that narrow miss.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Two parallel causal GRU blocks with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
class KeywordGRU(nn.Module):
    """Three compact GRUs with an ordered causal temporal-pyramid readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(4 * 144, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 4, 144, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        output = torch.cat((output_a[:, 0, :], output_b[:, 0, :]), dim=-1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
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
        outputs = torch.cat((outputs_a, outputs_b), dim=-1)
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
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
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, summary, count = state
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