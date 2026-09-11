MECHANISM: Seven-bin progressive trailing-frame pruning

HYPOTHESIS: Processing frames 1–27 in seven causal temporal bins will retain at least 85% validation accuracy while reducing execution from 28 to 27 recurrent steps and total inference MACs from 614,542,600 to approximately 592,816,330.

INTENDED_EDIT: Restore the qualified 45/45/46 recurrent backbone, resize the temporal readout to seven bins, and schedule only frames 1–27 for standard 32-frame inputs.

EVIDENCE: The 28-step seven-bin 45/45/46 design achieved 86.38% accuracy at 614,542,600 MACs; its 1.38-point margin supports the smallest additional trailing-boundary pruning probe while preserving the same backbone and readout.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 138, 8)
=======
class KeywordGRU(nn.Module):
    """Three balanced GRUs with a seven-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(7 * 136, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden_a = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 138, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 8, device=device, dtype=dtype)
=======
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 7, 136, device=device, dtype=dtype)
        bin_counts = torch.zeros(batch_size, 7, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=7)
        assignment = F.one_hot(bin_index[:, 0], num_classes=8).to(output.dtype)
=======
        bin_index = (count.to(dtype=torch.long) // 4).clamp(max=6)
        assignment = F.one_hot(bin_index[:, 0], num_classes=7).to(output.dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bin_indices = (positions // 4).clamp(max=7)
        assignments = F.one_hot(bin_indices, num_classes=8).to(outputs.dtype)
=======
        bin_indices = (positions // 4).clamp(max=6)
        assignments = F.one_hot(bin_indices, num_classes=7).to(outputs.dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(1, available_frames - 4))
>>>>>>> REPLACE