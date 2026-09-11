MECHANISM: Asymmetric temporal-pyramid block-diagonal recurrence

HYPOTHESIS: Reducing one GRU branch from 48 to 47 units will retain at least 85% validation accuracy because the four-bin temporal readout qualified with a 1.38-point margin, while lowering total inference MACs to approximately 760,857,920.

INTENDED_EDIT: Change the three recurrent branch widths from 48/48/48 to 47/48/48 and resize the temporal-bin state and classifier from 144 to 143 features.

EVIDENCE: The current 48/48/48 temporal-pyramid model achieved 86.38% accuracy at 769,881,600 MACs, substantially above the same-width global-summary model’s 84.91%; removing one unit from only one branch is the smallest structural capacity probe that preserves the successful ordered readout.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three compact GRUs with an ordered causal temporal-pyramid readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(4 * 144, 8)
=======
class KeywordGRU(nn.Module):
    """Three asymmetric GRUs with an ordered causal temporal-pyramid readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(4 * 143, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 4, 144, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 4, 143, device=device, dtype=dtype)
>>>>>>> REPLACE