MECHANISM: Continued recurrent-width reduction with salience-preserving temporal pooling

HYPOTHESIS: A 60-unit GRU using mean, maximum, and final recurrent outputs will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 10.7% versus the verified 64-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 64 to 60 and resize all recurrent state tensors and the three-way classifier input accordingly, preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 64-unit model achieved 86.01% accuracy, and every tested max-augmented width from 64 through 84 passed; its 1.01-point margin supports continuing the established four-unit structural width search.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean and final-state temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
=======
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean, maximum, and final temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(180, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 60, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 60, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 60, device=device, dtype=dtype)
>>>>>>> REPLACE