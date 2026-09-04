MECHANISM: Four-way block-diagonal gated recurrence

HYPOTHESIS: Four parallel 40-unit GRUs will retain at least 85% validation accuracy by increasing aggregate recurrent width from 145 to 160, while reducing expected total inference MACs from 776,225,560 to approximately 752,147,200.

INTENDED_EDIT: Replace the three 49-unit GRU branches with four 40-unit branches, concatenate their outputs into a 160-dimensional temporal mean, and preserve the full-frame training procedure.

EVIDENCE: Three 48-unit GRUs missed qualification by one example, while 49/48/48 qualified at 85.52%; four 40-unit blocks provide more aggregate units than either design while lowering the quadratic recurrent cost per step from 29,727 to 28,800 MACs.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three parallel causal GRU blocks with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 49, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 49, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 49, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(147, 8)
=======
class KeywordGRU(nn.Module):
    """Four parallel causal GRU blocks with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.gru_d = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(160, 8)
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
        hidden_b = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 147, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, summary, count
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
    ]:
        hidden_a = torch.zeros(batch_size, 1, 40, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 40, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 40, device=device, dtype=dtype)
        hidden_d = torch.zeros(batch_size, 1, 40, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 160, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, hidden_c, hidden_d, summary, count
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
        state: tuple[
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
    ]:
        hidden_a, hidden_b, hidden_c, hidden_d, summary, count = state
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
        output_d, hidden_d = self.gru_d(
            normalized, hidden_d.transpose(0, 1).contiguous()
        )
        output = torch.cat(
            (
                output_a[:, 0, :],
                output_b[:, 0, :],
                output_c[:, 0, :],
                output_d[:, 0, :],
            ),
            dim=-1,
        )
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            hidden_d.transpose(0, 1),
            summary + output,
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
        state: tuple[
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
    ]:
        hidden_a, hidden_b, hidden_c, hidden_d, summary, count = state
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
        outputs_d, hidden_d = self.gru_d(
            normalized, hidden_d.transpose(0, 1).contiguous()
        )
        outputs = torch.cat(
            (outputs_a, outputs_b, outputs_c, outputs_d), dim=-1
        )
        return (
            hidden_a.transpose(0, 1),
            hidden_b.transpose(0, 1),
            hidden_c.transpose(0, 1),
            hidden_d.transpose(0, 1),
            summary + outputs.sum(dim=1),
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
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, _hidden_c, _hidden_d, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
>>>>>>> REPLACE