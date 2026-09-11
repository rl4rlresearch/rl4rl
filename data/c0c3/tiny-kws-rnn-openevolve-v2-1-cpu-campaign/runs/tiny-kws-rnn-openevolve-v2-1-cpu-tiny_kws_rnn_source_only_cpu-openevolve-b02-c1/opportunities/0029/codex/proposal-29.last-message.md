MECHANISM: Single-GRU anchor with coupled-gate auxiliary recurrence

HYPOTHESIS: Retaining the dominant 60-unit GRU while replacing only the 59-unit branch with a coupled forget cell will preserve at least 85% validation accuracy and reduce total inference MACs from 671,511,100 to approximately 561,348,365.

INTENDED_EDIT: Replace the smaller GRU with a two-output `nn.Linear` recurrent cell while preserving the 60-unit GRU, 29-frame schedule, temporal mean summary, classifier, and training procedure.

EVIDENCE: Replacing both GRUs with coupled forget cells narrowly missed the target at 84.79% while cutting recurrent MACs by one third; retaining a full GRU in the larger branch tests whether one reset-gated recurrent path can recover that small accuracy deficit while still removing 110,162,735 recurrent MACs.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Asymmetric parallel causal GRUs with a shared online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 7)
=======
class CoupledForgetCell(nn.Module):
    """Single-gate recurrent cell with a coupled input gate."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.affine = nn.Linear(input_size + hidden_size, 2 * hidden_size)

    def forward(
        self, frame: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        forget_logits, candidate_logits = self.affine(
            torch.cat((frame, hidden), dim=1)
        ).chunk(2, dim=1)
        forget = torch.sigmoid(forget_logits)
        candidate = torch.tanh(candidate_logits)
        return forget * hidden + (1.0 - forget) * candidate


class KeywordGRU(nn.Module):
    """Full-GRU anchor with a cheaper coupled-gate auxiliary branch."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.cell_b = CoupledForgetCell(20, 59)
        self.classifier = nn.Linear(119, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        output_a, next_hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        next_hidden_b = self.cell_b(normalized[:, 0, :], hidden_b[:, 0, :])
        output = torch.cat((output_a[:, 0, :], next_hidden_b), dim=1)
        return (
            next_hidden_a.transpose(0, 1),
            next_hidden_b.unsqueeze(1),
            summary + output,
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        outputs_a, next_hidden_a = self.gru_a(
            normalized, hidden_a.transpose(0, 1).contiguous()
        )
        current_b = hidden_b[:, 0, :]
        outputs_b = []
        for normalized_frame in normalized.unbind(dim=1):
            current_b = self.cell_b(normalized_frame, current_b)
            outputs_b.append(current_b)
        stacked_b = torch.stack(outputs_b, dim=1)
        outputs = torch.cat((outputs_a, stacked_b), dim=2)
        return (
            next_hidden_a.transpose(0, 1),
            current_b.unsqueeze(1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )
>>>>>>> REPLACE