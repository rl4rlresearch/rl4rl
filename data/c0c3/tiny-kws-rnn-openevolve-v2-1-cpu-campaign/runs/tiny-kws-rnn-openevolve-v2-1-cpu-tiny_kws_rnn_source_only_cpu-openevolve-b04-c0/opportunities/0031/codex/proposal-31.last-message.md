MECHANISM: Complementary minimum-statistic temporal readout

HYPOTHESIS: A 64-unit GRU augmented with a running-minimum summary will recover validation accuracy to at least 85% while retaining approximately 2.6% fewer dense inference MACs than the verified 65-unit model.

INTENDED_EDIT: Reduce the GRU to 64 units and expand its temporal readout from mean/final/maximum to mean/final/maximum/minimum, producing a 256-feature classifier input while preserving all 32 causal steps and the training procedure.

EVIDENCE: The plain 64-unit model reached 84.54%, missing the threshold by only 0.46 points, while the 65-unit model reached 86.50%; adding a complementary extrema statistic costs only one additional 64-by-8 classifier slice and may recover the small accuracy deficit without restoring the substantially larger recurrent matrix.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean, final, and max temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(195, 8)
=======
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean, final, and extrema temporal readout."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 65), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, running_max, count
=======
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 64), -1.0, device=device, dtype=dtype
        )
        running_min = torch.full(
            (batch_size, 64), 1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, running_max, running_min, count
>>>>>>> REPLACE

<<<<<<< SEARCH
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, running_max, count = state
=======
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
        hidden, summary, running_max, running_min, count = state
>>>>>>> REPLACE

<<<<<<< SEARCH
            summary + output,
            torch.maximum(running_max, output),
            count + 1.0,
=======
            summary + output,
            torch.maximum(running_max, output),
            torch.minimum(running_min, output),
            count + 1.0,
>>>>>>> REPLACE

<<<<<<< SEARCH
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, running_max, count = state
=======
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
        hidden, summary, running_max, running_min, count = state
>>>>>>> REPLACE

<<<<<<< SEARCH
            summary + outputs.sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            count + frames.shape[1],
=======
            summary + outputs.sum(dim=1),
            torch.maximum(running_max, outputs.amax(dim=1)),
            torch.minimum(running_min, outputs.amin(dim=1)),
            count + frames.shape[1],
>>>>>>> REPLACE

<<<<<<< SEARCH
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, running_max, count = state
=======
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        hidden, summary, running_max, running_min, count = state
>>>>>>> REPLACE

<<<<<<< SEARCH
                hidden[:, 0, :],
                running_max,
            ),
=======
                hidden[:, 0, :],
                running_max,
                running_min,
            ),
>>>>>>> REPLACE