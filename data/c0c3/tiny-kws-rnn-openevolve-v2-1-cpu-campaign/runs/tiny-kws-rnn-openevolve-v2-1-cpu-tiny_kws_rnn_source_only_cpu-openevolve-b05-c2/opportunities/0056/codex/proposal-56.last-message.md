MECHANISM: Single-gate recurrent width reallocation

HYPOTHESIS: A 79-unit minimal gated recurrent unit on the qualified 21-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 269,773,150.

INTENDED_EDIT: Replace the three-affine 64-unit GRU with a two-affine 79-unit minimal gated recurrent update, preserve all four temporal summaries, and adopt the qualified 21-frame schedule.

EVIDENCE: The 63-unit GRU qualified at 85.03%, while the 112-unit ungated tanh RNN failed at 82.70%; this indicates gating remains important and motivates reallocating three-affine GRU computation into a wider single-gate state rather than removing gating entirely.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A causal GRU with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
class KeywordGRU(nn.Module):
    """A causal minimal gated unit with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.update_gate = nn.Linear(99, 79)
        self.candidate = nn.Linear(99, 79)
        self.classifier = nn.Linear(316, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 79, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 79, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, minimum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            torch.minimum(minimum, output),
            count + 1.0,
        )
=======
        hidden, summary, maximum, minimum, count = state
        hidden_vector = hidden[:, 0, :]
        normalized = self.input_norm(frame)
        gate = torch.sigmoid(
            self.update_gate(torch.cat((normalized, hidden_vector), dim=-1))
        )
        candidate = torch.tanh(
            self.candidate(
                torch.cat((normalized, gate * hidden_vector), dim=-1)
            )
        )
        output = (1.0 - gate) * hidden_vector + gate * candidate
        return (
            output.unsqueeze(1),
            summary + output,
            torch.maximum(maximum, output),
            torch.minimum(minimum, output),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, minimum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            torch.minimum(minimum, outputs.amin(dim=1)),
            count + frames.shape[1],
        )
=======
        for frame in frames.unbind(dim=1):
            state = self.recurrent_step(frame, state)
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE