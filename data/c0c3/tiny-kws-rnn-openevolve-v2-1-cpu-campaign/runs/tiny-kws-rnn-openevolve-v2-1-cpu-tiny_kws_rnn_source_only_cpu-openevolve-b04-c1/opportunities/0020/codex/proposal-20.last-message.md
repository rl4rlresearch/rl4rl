MECHANISM: Full-rank minimally gated recurrent update

HYPOTHESIS: A 76-unit, two-transform MGU will preserve at least 85% validation accuracy while reducing total dense inference MACs by about 9.4% versus the verified 64-unit three-transform GRU.

INTENDED_EDIT: Replace the GRU with a wider full-rank recurrent state using one state-conditioned update gate and one gated candidate transform, while retaining all 32 frames and mean/maximum/final pooling.

EVIDENCE: The 64-unit GRU reached 86.01% accuracy at 421.87M MACs, while 72- and 76-unit GRUs reached 87.24% and 86.38%. This supports trading the GRU’s third dense transform for wider state capacity. Unlike the failed low-rank residual model, this update retains full-rank recurrence and makes its gate depend on prior state.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean and final-state temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
=======
class KeywordMGU(nn.Module):
    """A full-rank minimally gated recurrent keyword spotter."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.update_gate = nn.Linear(96, 76)
        self.candidate = nn.Linear(96, 76)
        self.candidate_norm = nn.LayerNorm(76)
        self.classifier = nn.Linear(228, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return hidden.transpose(0, 1), summary + output, maximum, count + 1.0
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        normalized = self.input_norm(frame)
        update = torch.sigmoid(
            self.update_gate(torch.cat((normalized, hidden), dim=1))
        )
        candidate = torch.tanh(
            self.candidate_norm(
                self.candidate(
                    torch.cat((normalized, update * hidden), dim=1)
                )
            )
        )
        output = hidden + update * (candidate - hidden)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return output, summary + output, maximum, count + 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        sequence_maximum = outputs.amax(dim=1)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, sequence_maximum),
            sequence_maximum,
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            maximum,
            count + frames.shape[1],
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for frame in frames.unbind(dim=1):
            state = self.recurrent_step(frame, state)
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
=======
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordGRU()
=======
def build_model() -> nn.Module:
    return KeywordMGU()
>>>>>>> REPLACE