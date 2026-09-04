MECHANISM: Parallel block-diagonal gated recurrence

HYPOTHESIS: Two independent 64-unit GRUs processing all 32 frames will retain at least 85% validation accuracy while reducing expected total inference MACs to approximately 842,071,040, below the qualified 98-unit GRU’s 905,406,320.

INTENDED_EDIT: Replace the single 112-unit GRU with two parallel 64-unit GRUs, concatenate their outputs into a 128-dimensional temporal summary, and preserve the full-frame schedule and training procedure.

EVIDENCE: The full 128-unit GRU achieved 87.24% accuracy, while the 98-unit GRU exposed a capacity boundary at 85.03% and temporal subsampling failed below 83%. Parallel 64-unit recurrent blocks retain 128 aggregate gated units and every input frame while structurally reducing quadratic hidden-to-hidden MACs by about 7% versus the qualified 98-unit model.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
=======
class KeywordGRU(nn.Module):
    """Two parallel causal GRU blocks with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_a = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden_a, hidden_b, summary, count
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        _hidden_a, _hidden_b, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
>>>>>>> REPLACE