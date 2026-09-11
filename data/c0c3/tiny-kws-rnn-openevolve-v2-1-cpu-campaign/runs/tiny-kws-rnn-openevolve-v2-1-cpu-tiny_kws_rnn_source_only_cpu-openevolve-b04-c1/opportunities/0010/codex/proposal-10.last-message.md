MECHANISM: Input-gated low-rank residual recurrence

HYPOTHESIS: A 128-dimensional residual state with an input-conditioned update gate and rank-64 learned recurrent transform will retain at least 85% accuracy while reducing dense MACs by roughly 30% versus the verified 92-unit GRU.

INTENDED_EDIT: Replace the three-gate full-rank GRU with a residual recurrent update using one input projection and a 128→64→128 recurrent bottleneck, while preserving all 32 frames and the successful mean-plus-final readout.

EVIDENCE: The 92-unit GRU achieved 87.12% accuracy, leaving 2.12 points of headroom, but spends nearly all 807.4M MACs on three full-rank gated transforms. Prior frame decimation harmed accuracy, so this patch preserves temporal resolution and instead challenges the assumption that every gate requires a full-rank recurrent matrix.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean and final-state temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)
=======
class KeywordGRU(nn.Module):
    """A causal input-gated residual RNN with low-rank state mixing."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 256)
        self.recurrent_down = nn.Linear(128, 64, bias=False)
        self.recurrent_up = nn.Linear(64, 128, bias=False)
        self.classifier = nn.Linear(256, 8)
        with torch.no_grad():
            self.input_projection.bias[128:].copy_(
                torch.linspace(-2.0, 1.0, 128)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        hidden, summary, count = state
        previous = hidden[:, 0, :]
        drive, gate_logits = self.input_projection(
            self.input_norm(frame)
        ).chunk(2, dim=1)
        recurrent = self.recurrent_up(
            torch.tanh(self.recurrent_down(previous))
        )
        candidate = torch.tanh(drive + recurrent)
        gate = torch.sigmoid(gate_logits)
        output = previous + gate * (candidate - previous)
        return output.unsqueeze(1), summary + output, count + 1.0
>>>>>>> REPLACE