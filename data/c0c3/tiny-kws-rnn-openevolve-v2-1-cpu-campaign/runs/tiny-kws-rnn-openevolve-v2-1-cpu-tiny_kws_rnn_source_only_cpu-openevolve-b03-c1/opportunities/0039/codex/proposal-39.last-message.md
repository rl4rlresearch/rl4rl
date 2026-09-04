MECHANISM: Asymmetric single-head width reduction

HYPOTHESIS: Two 32-unit GRU heads plus one 31-unit head will retain at least 85% validation accuracy while reducing total inference MACs versus the verified three-by-32 model.

INTENDED_EDIT: Reduce the third GRU head from 32 to 31 units and resize its hidden state, aggregate summaries, and classifier input from 96/288 to 95/285 features.

EVIDENCE: Three 32-unit heads achieved 86.01% accuracy, leaving 1.01 percentage points of margin; prior asymmetric one-unit reductions from 33/33/33 through 33/32/32 all retained the target, making a single-head reduction the smallest evidence-backed structural change.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three compact parallel causal GRU heads with equal widths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(288, 8)
=======
class KeywordGRU(nn.Module):
    """Three compact parallel causal GRU heads with asymmetric widths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 31, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(285, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_0 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden_0 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 31, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 95, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 95, device=device, dtype=dtype)
>>>>>>> REPLACE