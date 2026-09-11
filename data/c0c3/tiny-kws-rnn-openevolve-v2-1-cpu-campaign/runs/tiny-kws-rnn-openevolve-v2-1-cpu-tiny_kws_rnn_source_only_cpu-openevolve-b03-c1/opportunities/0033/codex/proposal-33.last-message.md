MECHANISM: Three-head recurrent width boundary reduction

HYPOTHESIS: Three independent 35-unit GRU heads will retain at least 85% validation accuracy while reducing total inference MACs by approximately 4.5% versus the verified three-by-36 model.

INTENDED_EDIT: Reduce each GRU head from 36 to 35 units and resize the hidden state, trajectory summaries, and classifier input from 108/324 to 105/315 features.

EVIDENCE: Three 36-unit heads achieved 86.01%, while four 29-unit heads fell to 84.29% despite greater aggregate width; preserving the successful three-head topology and removing only one unit per head is the most direct capacity-boundary test.

<<<<<<< SEARCH
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
=======
class KeywordGRU(nn.Module):
    """Three compact parallel causal GRU heads with complementary dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_heads = nn.ModuleList(
            [
                nn.GRU(20, 35, num_layers=1, batch_first=True),
                nn.GRU(20, 35, num_layers=1, batch_first=True),
                nn.GRU(20, 35, num_layers=1, batch_first=True),
            ]
        )
        self.classifier = nn.Linear(315, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 3, 36, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 108, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 3, 35, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 105, device=device, dtype=dtype)
        peak = torch.zeros(batch_size, 105, device=device, dtype=dtype)
>>>>>>> REPLACE