MECHANISM: Asymmetric three-way block-diagonal GRU with eight-bin temporal readout

HYPOTHESIS: Reducing one recurrent branch from 47 to 46 units will retain at least 85% validation accuracy while lowering total inference MACs from 746,487,840 to approximately 737,594,560.

INTENDED_EDIT: Change the three recurrent widths from 47/47/47 to 46/47/47 and resize the temporal-bin state and classifier from 141 to 140 features.

EVIDENCE: The 47/47/48 eight-bin design achieved 86.13% accuracy, and the current 47/47/47 design still achieved 85.77%; removing one unit from only one branch is the smallest structural capacity probe and should save about 8.9 million MACs.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 141, 8)
=======
class KeywordGRU(nn.Module):
    """Three asymmetric GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 140, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 141, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 140, device=device, dtype=dtype)
>>>>>>> REPLACE