MECHANISM: Wide-state low-rank gated recurrence

HYPOTHESIS: A persistent 128-unit gated state with a learned rank-32 recurrent bottleneck will retain at least 85% accuracy on the 28-frame schedule while reducing total dense inference MACs by approximately 8% versus the verified 84-unit GRU.

INTENDED_EDIT: Replace the full-rank GRU with a custom GRU-like update whose input transformation remains wide but whose recurrent gate transformation passes through a shared 32-dimensional learned bottleneck; preserve the mean-plus-terminal readout and endpoint-preserving schedule.

EVIDENCE: The 84-unit, 28-step GRU passed at 85.64% accuracy and 599.16M MACs, while the failed reset-based hierarchy reached only 82.94%, indicating that persistent state and retained observations are load-bearing. This patch preserves both while directly testing the shared assumption that every gate needs an independent full-rank hidden-to-hidden matrix.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with mean and terminal-state readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
class BottleneckKeywordGRU(nn.Module):
    """A wide gated recurrent state with low-rank recurrent transforms."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_to_gates = nn.Linear(20, 384)
        self.recurrent_down = nn.Linear(128, 32, bias=False)
        self.recurrent_to_gates = nn.Linear(32, 384, bias=False)
        self.classifier = nn.Linear(256, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
        hidden = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0
=======
        hidden, summary, count = state
        input_reset, input_update, input_candidate = self.input_to_gates(
            self.input_norm(frame)
        ).chunk(3, dim=-1)
        recurrent_code = torch.tanh(self.recurrent_down(hidden))
        recurrent_reset, recurrent_update, recurrent_candidate = (
            self.recurrent_to_gates(recurrent_code).chunk(3, dim=-1)
        )
        reset = torch.sigmoid(input_reset + recurrent_reset)
        update = torch.sigmoid(input_update + recurrent_update)
        candidate = torch.tanh(
            input_candidate + reset * recurrent_candidate
        )
        hidden = update * hidden + (1.0 - update) * candidate
        return hidden, summary + hidden, count + 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden), dim=-1))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordGRU()
=======
def build_model() -> nn.Module:
    return BottleneckKeywordGRU()
>>>>>>> REPLACE