MECHANISM: Fixed adjacent-band spectral pooling

HYPOTHESIS: Merging the two highest adjacent mel bands will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.85% versus the verified 98-unit model.

INTENDED_EDIT: Preserve the 98-unit GRU and all 32 causal steps, but average the final two mel bands into one feature and reduce the GRU input width from 20 to 19.

EVIDENCE: The 98-unit full-frame model passed at 85.03%, while reducing hidden width to 97 and reducing temporal coverage both failed; this motivates an orthogonal, minimal reduction in redundant adjacent spectral resolution.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
=======
class KeywordGRU(nn.Module):
    """A causal GRU using fixed pooling of two adjacent spectral bands."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)

    @staticmethod
    def compress_bands(frames: torch.Tensor) -> torch.Tensor:
        pooled_high_band = frames[..., 18:].mean(dim=-1, keepdim=True)
        return torch.cat((frames[..., :18], pooled_high_band), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self.input_norm(self.compress_bands(frame)).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self.input_norm(self.compress_bands(frames)),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE