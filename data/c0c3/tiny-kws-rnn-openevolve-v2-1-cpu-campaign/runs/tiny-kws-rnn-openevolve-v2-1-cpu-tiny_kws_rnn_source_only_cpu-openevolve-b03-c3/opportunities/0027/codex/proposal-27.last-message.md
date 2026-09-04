MECHANISM: Balanced three-way block-diagonal GRU with eight-bin temporal readout

HYPOTHESIS: Three 46-unit GRUs with eight ordered temporal bins will retain at least 85% validation accuracy while reducing total inference MACs from 728,701,280 to approximately 719,808,000.

INTENDED_EDIT: Reduce the remaining 47-unit branch in the qualified 46/46/47 design to 46 units and resize its eight-bin state and classifier to 138 features.

EVIDENCE: The 46/46/47 eight-bin model achieved 86.38% accuracy, improving upon 46/47/47 despite its lower recurrent capacity; the 1.38-point margin supports the smallest balanced width reduction.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three asymmetric GRUs with an ordered causal temporal-pyramid readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(4 * 143, 8)
=======
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 138, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 4, 143, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 4, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 138, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 8, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bin_index = (count.to(dtype=torch.long) // 8).clamp(max=3)
        assignment = F.one_hot(bin_index[:, 0], num_classes=4).to(output.dtype)
=======
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=7)
        assignment = F.one_hot(bin_index[:, 0], num_classes=8).to(output.dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bin_indices = (positions // 8).clamp(max=3)
        assignments = F.one_hot(bin_indices, num_classes=4).to(outputs.dtype)
=======
        bin_indices = (positions // 4).clamp(max=7)
        assignments = F.one_hot(bin_indices, num_classes=8).to(outputs.dtype)
>>>>>>> REPLACE