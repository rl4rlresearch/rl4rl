MECHANISM: Width reduction with dual-timescale recurrent readout

HYPOTHESIS: A 100-unit GRU using concatenated mean and final recurrent outputs will retain at least 85% validation accuracy while reducing total inference MACs by approximately 6.9% versus the verified 104-unit model.

INTENDED_EDIT: Reduce recurrent width from 104 to 100, preserve all 32 frames, and classify from both the sequence-average output and final hidden output.

EVIDENCE: The 104-unit full-resolution GRU achieved 85.03% accuracy; retaining every frame while enriching its temporal readout may compensate for a modest width reduction, which structurally lowers recurrent MACs.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)
=======
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean and final-state temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(200, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)
>>>>>>> REPLACE