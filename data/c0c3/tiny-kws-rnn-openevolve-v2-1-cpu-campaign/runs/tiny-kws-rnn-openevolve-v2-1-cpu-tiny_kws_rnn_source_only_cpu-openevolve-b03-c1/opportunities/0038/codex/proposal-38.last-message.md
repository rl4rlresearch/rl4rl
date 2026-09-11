MECHANISM: Symmetric three-head width boundary reduction

HYPOTHESIS: Three independent 32-unit GRU heads will retain at least 85% validation accuracy while reducing total inference MACs versus the verified 33/32/32 model.

INTENDED_EDIT: Reduce the remaining 33-unit GRU head to 32 units and resize the hidden state, aggregate summaries, and classifier input from 97/291 to 96/288 features.

EVIDENCE: The verified 33/32/32 model achieved 85.40% accuracy, and each preceding single-head width reduction retained the target; this is the smallest structural reduction that tests the next capacity boundary.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three compact parallel causal GRU heads with asymmetric widths."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 33, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
                nn.GRU(20, 32, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(291, 8)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_0 = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 97, device=device, dtype=dtype)
=======
        hidden_0 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_1 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        hidden_2 = torch.zeros(batch_size, 32, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE