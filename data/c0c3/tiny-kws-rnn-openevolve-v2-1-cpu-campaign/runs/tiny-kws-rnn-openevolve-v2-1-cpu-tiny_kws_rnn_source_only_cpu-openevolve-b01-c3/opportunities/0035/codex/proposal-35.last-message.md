MECHANISM: Full-rank gate feedback with a narrower recurrent state

HYPOTHESIS: A 102-unit gated state retaining all three 32-unit feedback bottlenecks will achieve at least 85% validation accuracy while reducing total inference MACs from 339,809,360 to approximately 336,510,240.

INTENDED_EDIT: Replace the dense 93-unit GRU with a 102-unit bottleneck-gated recurrence, preserving the proven 16-frame schedule and mean/final-state readout.

EVIDENCE: The 103-state design with three 32-unit bottlenecks achieved 85.399% accuracy at 339,809,360 MACs, while bottleneck-rank reductions were less reliable; trimming state width by one is the closest untested structural reduction that preserves full gate-feedback capacity.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 93, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(186, 8)
=======
class KeywordGRU(nn.Module):
    """A gated recurrent memory with low-dimensional feedback paths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 306)
        self.recurrent_down = nn.Linear(102, 96)
        self.recurrent_up = nn.ModuleList(
            [nn.Linear(32, 102) for _ in range(3)]
        )
        self.classifier = nn.Linear(204, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 93, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 93, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 102, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 102, device=device, dtype=dtype)
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