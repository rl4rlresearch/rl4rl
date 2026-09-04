MECHANISM: Parallel block-diagonal recurrent ensemble

HYPOTHESIS: Two independent 48-unit GRU heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 12.5% versus the verified 77-unit GRU.

INTENDED_EDIT: Replace the single densely coupled recurrent state with two continuous 48-unit GRU states that independently encode every scheduled frame, then fuse their endpoints, temporal means, and temporal maxima for classification.

EVIDENCE: The 77-unit model achieved 86.38%, providing accuracy margin, while its dense hidden-to-hidden matrix dominates cost. Unlike the failed chunk hierarchy, this design never resets local memory or compresses chunks through an intermittently updated bottleneck; both heads preserve the proven 27-step schedule and readout.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 77, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(231, 8)
=======
class KeywordGRU(nn.Module):
    """Two parallel causal GRU heads with complementary learned dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 48, num_layers=1, batch_first=True),
                nn.GRU(20, 48, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(288, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 77, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 77, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 77, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 2, 48, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, peak, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )
=======
        hidden, summary, peak, count = state
        normalized = self.input_norm(frame).unsqueeze(1)
        outputs = []
        next_hidden = []
        for head_index, gru in enumerate(self.gru_heads):
            head_output, head_hidden = gru(
                normalized,
                hidden[:, head_index, :].unsqueeze(0).contiguous(),
            )
            outputs.append(head_output[:, 0, :])
            next_hidden.append(head_hidden[0])
        output = torch.cat(outputs, dim=-1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + output,
            torch.maximum(peak, output),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, peak, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
=======
        hidden, summary, peak, count = state
        normalized = self.input_norm(frames)
        head_outputs = []
        next_hidden = []
        for head_index, gru in enumerate(self.gru_heads):
            output, head_hidden = gru(
                normalized,
                hidden[:, head_index, :].unsqueeze(0).contiguous(),
            )
            head_outputs.append(output)
            next_hidden.append(head_hidden[0])
        outputs = torch.cat(head_outputs, dim=-1)
        return (
            torch.stack(next_hidden, dim=1),
            summary + outputs.sum(dim=1),
            torch.maximum(peak, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat((hidden[:, 0, :], mean_output, peak), dim=-1)
        )
=======
        return self.classifier(
            torch.cat((hidden.flatten(start_dim=1), mean_output, peak), dim=-1)
        )
>>>>>>> REPLACE