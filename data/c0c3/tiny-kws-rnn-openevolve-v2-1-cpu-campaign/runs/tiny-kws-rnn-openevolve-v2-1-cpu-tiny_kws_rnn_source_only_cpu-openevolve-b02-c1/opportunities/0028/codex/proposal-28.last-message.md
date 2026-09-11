MECHANISM: Coupled single-gate recurrent cells

HYPOTHESIS: Replacing each three-gate GRU with a width-matched coupled forget cell will retain at least 85% validation accuracy while reducing recurrent MACs by exactly one third, from 670,832,205 to 447,221,470, with the same 29 recurrent steps.

INTENDED_EDIT: Preserve the verified 60+59 hidden widths, causal mean summary, frame schedule, and orthonormal classifier, but replace each GRU with a two-output-per-hidden-unit `nn.Linear` recurrent cell.

EVIDENCE: The 100+100 ungated tanh RNN fell to 71.78% while the 60+59 GRU reached 85.40%, indicating gating is load-bearing. A coupled forget cell retains learned elementwise gating and recurrent dependence while removing one third of the GRU’s dense gate computation.

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
class CoupledForgetCell(nn.Module):
    """One-gate recurrent cell using two affine outputs per hidden unit."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.affine = nn.Linear(input_size + hidden_size, 2 * hidden_size)

    def forward(
        self, frame: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        forget_logits, proposal_logits = self.affine(
            torch.cat((frame, hidden), dim=1)
        ).chunk(2, dim=1)
        forget = torch.sigmoid(forget_logits)
        proposal = torch.tanh(proposal_logits)
        return forget * hidden + (1.0 - forget) * proposal


class KeywordGRU(nn.Module):
    """Asymmetric parallel coupled-gate cells with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.cell_a = CoupledForgetCell(20, 60)
        self.cell_b = CoupledForgetCell(20, 59)
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
        normalized = self.input_norm(frame)
        output_a = self.cell_a(normalized, hidden_a[:, 0, :])
        output_b = self.cell_b(normalized, hidden_b[:, 0, :])
        output = torch.cat((output_a, output_b), dim=1)
        return (
            output_a.unsqueeze(1),
            output_b.unsqueeze(1),
            summary + output,
            count + 1.0,
        )

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for frame_index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, frame_index, :], state)
        return state

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
>>>>>>> REPLACE