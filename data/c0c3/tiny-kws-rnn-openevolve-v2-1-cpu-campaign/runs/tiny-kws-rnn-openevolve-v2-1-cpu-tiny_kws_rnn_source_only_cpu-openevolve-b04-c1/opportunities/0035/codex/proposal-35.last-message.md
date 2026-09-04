MECHANISM: Capacity-for-time recurrent tradeoff

HYPOTHESIS: A 59-unit GRU processing frames 3–26 will retain at least 85% validation accuracy while executing 24 recurrent steps and fewer dense MACs than the verified 58-unit, 25-step model.

INTENDED_EDIT: Increase GRU width from 58 to 59 units, resize its state and classifier, and remove the latest frame from the successful 3–27 window.

EVIDENCE: The 58-unit model passed with frames 3–27 but failed with frames 4–28, indicating earlier frames are more valuable at this boundary; one added recurrent unit provides a modest capacity buffer while the 24-step design still reduces recurrent MACs by approximately 1.1%.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean, maximum, and final temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
=======
class KeywordGRU(nn.Module):
    """A compact 24-step causal GRU with pooled temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(177, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 59, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 59, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 59, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(25, available_frames)
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(24, available_frames)
        start = max(0, (available_frames - steps) // 2 - 1)
        return list(range(start, start + steps))
>>>>>>> REPLACE