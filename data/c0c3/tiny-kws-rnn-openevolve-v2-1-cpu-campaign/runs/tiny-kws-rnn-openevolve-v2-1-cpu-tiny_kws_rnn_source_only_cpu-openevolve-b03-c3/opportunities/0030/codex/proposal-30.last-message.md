MECHANISM: Two-timescale hierarchical GRU with quarter-rate semantic updates

HYPOTHESIS: A 40-unit fast GRU feeding a 72-unit GRU every fourth frame will retain at least 85% accuracy while reducing expected total inference MACs from 702,334,400 to approximately 351,349,760.

INTENDED_EDIT: Replace three independent every-frame GRUs with one fast acoustic GRU and a causally coupled slow semantic GRU; classify eight boundary readouts containing both timescales.

EVIDENCE: The 45/45/46 model reached 87.61% accuracy, but its three same-rate branches spend 695,240,640 recurrent MACs processing redundant inputs. The failed four-branch repartition showed that parallel width alone is insufficient, motivating learned cross-timescale composition instead.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 138, 8)
=======
class KeywordGRU(nn.Module):
    """Fast acoustic recurrence with a quarter-rate semantic recurrence."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_gru = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.slow_gru = nn.GRU(40, 72, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 138, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 8, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, bins, bin_counts, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        fast_hidden = torch.zeros(batch_size, 1, 40, device=device, dtype=dtype)
        slow_hidden = torch.zeros(batch_size, 1, 72, device=device, dtype=dtype)
        readouts = torch.zeros(batch_size, 8, 112, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return fast_hidden, slow_hidden, readouts, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        fast_hidden, slow_hidden, readouts, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        fast_output, fast_hidden = self.fast_gru(
            normalized, fast_hidden.transpose(0, 1).contiguous()
        )
        fast_hidden = fast_hidden.transpose(0, 1)

        position = int(count[0, 0].item())
        if position % 4 != 3:
            return fast_hidden, slow_hidden, readouts, count + 1.0

        slow_output, slow_hidden = self.slow_gru(
            fast_output, slow_hidden.transpose(0, 1).contiguous()
        )
        boundary_readout = torch.cat(
            (fast_output[:, 0, :], slow_output[:, 0, :]), dim=-1
        )
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=7)
        assignment = F.one_hot(
            bin_index[:, 0], num_classes=8
        ).to(boundary_readout.dtype)
        return (
            fast_hidden,
            slow_hidden.transpose(0, 1),
            readouts
            + assignment.unsqueeze(-1) * boundary_readout.unsqueeze(1),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        fast_hidden, slow_hidden, readouts, count = state
        normalized = self.input_norm(frames)
        fast_outputs, fast_hidden = self.fast_gru(
            normalized, fast_hidden.transpose(0, 1).contiguous()
        )
        fast_hidden = fast_hidden.transpose(0, 1)

        start = int(count[0, 0].item())
        first_boundary = (3 - start % 4) % 4
        boundary_offsets = torch.arange(
            first_boundary, frames.shape[1], 4, device=frames.device
        )
        if boundary_offsets.numel() == 0:
            return (
                fast_hidden,
                slow_hidden,
                readouts,
                count + frames.shape[1],
            )

        slow_inputs = fast_outputs.index_select(1, boundary_offsets)
        slow_outputs, slow_hidden = self.slow_gru(
            slow_inputs, slow_hidden.transpose(0, 1).contiguous()
        )
        boundary_readouts = torch.cat((slow_inputs, slow_outputs), dim=-1)
        positions = count.to(dtype=torch.long) + boundary_offsets.view(1, -1)
        bin_indices = (positions // 4).clamp(max=7)
        assignments = F.one_hot(
            bin_indices, num_classes=8
        ).to(boundary_readouts.dtype)
        readout_sums = (
            boundary_readouts.unsqueeze(2) * assignments.unsqueeze(-1)
        ).sum(dim=1)
        return (
            fast_hidden,
            slow_hidden.transpose(0, 1),
            readouts + readout_sums,
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, bins, bin_counts, _count = state
        pooled = bins / bin_counts.clamp_min(1.0).unsqueeze(-1)
        return self.classifier(pooled.flatten(start_dim=1))
=======
    def classify(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _fast_hidden, _slow_hidden, readouts, _count = state
        return self.classifier(readouts.flatten(start_dim=1))
>>>>>>> REPLACE