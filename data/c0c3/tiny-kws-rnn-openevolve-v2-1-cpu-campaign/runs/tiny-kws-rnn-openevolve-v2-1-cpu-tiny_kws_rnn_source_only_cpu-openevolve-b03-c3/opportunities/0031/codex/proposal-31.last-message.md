MECHANISM: Balanced three-way block-diagonal GRU with eight-bin temporal readout

HYPOTHESIS: Three 45-unit GRUs will retain at least 85% validation accuracy while reducing total inference MACs from 702,334,400 to approximately 693,597,600.

INTENDED_EDIT: Reduce all recurrent branches to 45 units and resize the aggregate temporal state and classifier to 135 features.

EVIDENCE: The qualified 45/45/46 model achieved 87.61% accuracy at 702,334,400 MACs, the strongest observed margin; reducing its remaining 46-unit branch is the smallest further structural cost probe.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three asymmetric GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 139, 8)
=======
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 135, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 139, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 135, device=device, dtype=dtype)
>>>>>>> REPLACE