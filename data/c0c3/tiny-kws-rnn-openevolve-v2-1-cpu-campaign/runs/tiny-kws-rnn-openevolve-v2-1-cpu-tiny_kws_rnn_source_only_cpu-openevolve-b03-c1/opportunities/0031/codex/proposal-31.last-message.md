MECHANISM: Three-way block-diagonal recurrent ensemble

HYPOTHESIS: Three independent 36-unit GRU heads will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.4% versus two 48-unit heads, because the aggregate recurrent representation grows from 96 to 108 features while each costly hidden-to-hidden matrix becomes smaller.

INTENDED_EDIT: Replace the two 48-unit GRU heads with three 36-unit heads and resize the state summaries and endpoint/mean/maximum classifier input accordingly.

EVIDENCE: Two independent 48-unit heads achieved 86.13% accuracy and substantially outperformed the denser 77-unit GRU on MACs, showing that partitioned recurrent state preserves accuracy; a third smaller block tests whether further structural factorization can exploit the same effect with more aggregate features and fewer MACs.

<<<<<<< SEARCH
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
=======
class KeywordGRU(nn.Module):
    """Three parallel causal GRU heads with complementary learned dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 36, num_layers=1, batch_first=True),
                nn.GRU(20, 36, num_layers=1, batch_first=True),
                nn.GRU(20, 36, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(324, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 2, 48, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 3, 36, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 108, device=device, dtype=dtype)
>>>>>>> REPLACE