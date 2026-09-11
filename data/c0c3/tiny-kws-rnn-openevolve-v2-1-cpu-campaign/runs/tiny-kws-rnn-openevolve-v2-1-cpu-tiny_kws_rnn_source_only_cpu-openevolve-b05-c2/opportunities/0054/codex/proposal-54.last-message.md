MECHANISM: Gate-to-width recurrent reallocation

HYPOTHESIS: A 112-unit tanh RNN using the qualified 22-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 267,998,080.

INTENDED_EDIT: Replace the 64-unit GRU with a wider, single-affine 112-unit tanh RNN, resize its summaries and classifier, and restore frame 6 to use the stronger 22-frame schedule.

EVIDENCE: The 22-step GRU achieved 86.01% accuracy, while the 63-unit 21-step GRU barely qualified at 85.03% and further width/readout contractions failed; reallocating gated-computation MACs into a wider recurrent state tests a structurally different efficiency frontier while recovering the stronger schedule.

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
    """A causal tanh RNN with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.RNN(
            20, 112, num_layers=1, nonlinearity="tanh", batch_first=True
        )
        self.classifier = nn.Linear(448, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 112, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
=======
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
>>>>>>> REPLACE