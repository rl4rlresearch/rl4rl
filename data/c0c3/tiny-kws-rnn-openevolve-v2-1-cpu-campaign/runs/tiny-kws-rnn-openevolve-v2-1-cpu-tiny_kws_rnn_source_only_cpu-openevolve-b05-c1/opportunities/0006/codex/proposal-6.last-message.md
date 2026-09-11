MECHANISM: Dual-timescale mean-and-terminal recurrent readout

HYPOTHESIS: A 91-unit GRU using both the temporal mean and final recurrent output will retain at least 85% validation accuracy while reducing total dense inference MACs by approximately 1.9% versus the verified 92-unit model.

INTENDED_EDIT: Reduce recurrent width from 92 to 91, preserve all 32 causal steps, and compensate for the smaller state by classifying the concatenated mean and terminal GRU outputs.

EVIDENCE: The 92-unit full-frame model narrowly passed at 85.03%, while temporal subsampling failed at 16 and 20 steps; preserving every frame and adding a richer, low-cost readout is therefore safer than removing temporal observations.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)
=======
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with mean and terminal-state readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(182, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))
>>>>>>> REPLACE