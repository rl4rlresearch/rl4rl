MECHANISM: Gate-bottleneck rank trim

HYPOTHESIS: Reducing each recurrent feedback bottleneck from 32 to 31 units while preserving the 104-unit state and 16-frame schedule will retain at least 85% accuracy and reduce total inference MACs from 343,108,480 to approximately 334,971,520.

INTENDED_EDIT: Replace the dense 96-unit GRU with the qualified gated bottleneck recurrence, using a 93-unit shared projection split into three independent 31-unit feedback paths.

EVIDENCE: The 104-state, 32-unit-per-gate bottleneck design achieved 85.521% accuracy at 343,108,480 MACs; trimming one coordinate per gate is the nearest untested structural reduction and preserves its demonstrated gating, state capacity, temporal coverage, and readout.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)
=======
class KeywordGRU(nn.Module):
    """A gated recurrent memory with reduced feedback bottlenecks."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 312)
        self.recurrent_down = nn.Linear(104, 93)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(31, 104) for _ in range(3)]
        )
        self.classifier = nn.Linear(208, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
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
        hidden = hidden[:, 0, :]

        input_reset, input_update, input_candidate = self.input_projection(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.tanh(
            self.recurrent_down(hidden)
        ).chunk(3, dim=1)

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return hidden.unsqueeze(1), summary + hidden, count + 1.0
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