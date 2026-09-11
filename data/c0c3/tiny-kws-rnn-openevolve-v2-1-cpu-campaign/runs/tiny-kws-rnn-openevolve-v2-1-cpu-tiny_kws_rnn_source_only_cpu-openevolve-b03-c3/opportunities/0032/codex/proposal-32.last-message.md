MECHANISM: Boundary-frame pruning with a high-margin recurrent backbone

HYPOTHESIS: The qualified 45/45/46 GRU model can omit the first input frame while retaining at least 85% validation accuracy, reducing execution to 31 recurrent steps and approximately 680,608,130 total MACs.

INTENDED_EDIT: Restore the high-margin 45/45/46 eight-bin model and causally process frames 1–31 instead of all 32 frames.

EVIDENCE: The 45/45/46 design achieved 87.61% accuracy at 702,334,400 MACs—the strongest observed margin—so removing one boundary frame is a conservative step-reduction probe with more accuracy headroom than the 85.89% three-45-unit model.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Three asymmetric GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 47, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 140, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 47, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 140, device=device, dtype=dtype)
=======
class KeywordGRU(nn.Module):
    """Three balanced GRUs with an eight-bin causal temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 45, num_layers=1, batch_first=True)
        self.gru_c = nn.GRU(20, 46, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(8 * 136, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        hidden_a = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_b = torch.zeros(batch_size, 1, 45, device=device, dtype=dtype)
        hidden_c = torch.zeros(batch_size, 1, 46, device=device, dtype=dtype)
        bins = torch.zeros(batch_size, 8, 136, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
>>>>>>> REPLACE