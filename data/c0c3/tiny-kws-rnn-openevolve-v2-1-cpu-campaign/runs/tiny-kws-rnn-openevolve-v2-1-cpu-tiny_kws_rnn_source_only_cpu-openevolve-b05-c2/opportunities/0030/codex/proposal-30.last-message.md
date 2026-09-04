MECHANISM: Adjacent-width GRU pruning with signed temporal-extrema readout

HYPOTHESIS: A 64-unit full-resolution GRU using mean, final, maximum, and minimum summaries will retain at least 85% validation accuracy while reducing total inference MACs by approximately 2.69% versus the qualified 65-unit design.

INTENDED_EDIT: Reduce the recurrent width from 67 to 64, add temporal-minimum tracking, and resize the classifier to consume four 64-unit summaries while preserving all 32 causal steps and the established training procedure.

EVIDENCE: The 65-unit signed-extrema design achieved 85.77% accuracy at 433,971,200 MACs, whereas the 65-unit triple-readout design failed; this isolates minimum pooling as useful at the width boundary and motivates testing the adjacent 64-unit signed-extrema model.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(201, 8)
=======
class KeywordGRU(nn.Module):
    """A causal GRU with signed temporal-extrema summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, minimum, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden, summary, maximum, minimum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            torch.minimum(minimum, output),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden, summary, maximum, minimum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            torch.minimum(minimum, outputs.amin(dim=1)),
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(
            torch.cat((mean_output, final_output, maximum), dim=-1)
        )
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, minimum, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(
            torch.cat(
                (mean_output, final_output, maximum, minimum),
                dim=-1,
            )
        )
>>>>>>> REPLACE