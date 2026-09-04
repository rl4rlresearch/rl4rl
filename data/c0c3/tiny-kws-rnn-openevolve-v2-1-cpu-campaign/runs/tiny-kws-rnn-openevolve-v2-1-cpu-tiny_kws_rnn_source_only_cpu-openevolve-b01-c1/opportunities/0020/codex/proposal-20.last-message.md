MECHANISM: Nonlinear segmented recurrent memory pooling

HYPOTHESIS: Seven causal four-frame memory segments with learned frame embeddings and sum/max statistics will retain at least 85% accuracy while reducing total inference MACs by roughly 55% versus the 28-step GRU.

INTENDED_EDIT: Replace dense hidden-to-hidden GRU updates with a two-layer per-frame encoder that recurrently accumulates coarse temporal sum and maximum memories, then predicts from all ordered segments with a small nonlinear head.

EVIDENCE: The 92-unit GRU remained accurate through 28 steps but failed only when a fifth leading frame was removed, suggesting that preserving temporal coverage matters more than repeatedly applying three dense recurrent gates; the new design keeps the verified 28-frame schedule while challenging that gate-heavy assumption.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return hidden.transpose(0, 1), summary + output, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))
=======
class SegmentMemoryNet(nn.Module):
    """A causal recurrent bank of ordered, nonlinear temporal summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.frame_encoder = nn.Sequential(
            nn.Linear(20, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.memory_norm = nn.LayerNorm(7 * 64 * 2)
        self.classifier = nn.Sequential(
            nn.Linear(7 * 64 * 2, 96),
            nn.SiLU(),
            nn.Linear(96, 8),
        )

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (batch_size, 7, 64)
        sums = torch.zeros(shape, device=device, dtype=dtype)
        maxima = torch.zeros(shape, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return sums, maxima, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sums, maxima, count = state
        encoded = self.frame_encoder(self.input_norm(frame))
        slot = (count.to(dtype=torch.long) // 4).clamp_max(6).squeeze(1)
        selector = F.one_hot(slot, num_classes=7).to(encoded.dtype).unsqueeze(-1)
        candidate = selector * encoded.unsqueeze(1)
        return sums + candidate, torch.maximum(maxima, candidate), count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sums, maxima, count = state
        encoded = self.frame_encoder(self.input_norm(frames))
        offsets = torch.arange(
            frames.shape[1], device=frames.device, dtype=torch.long
        ).unsqueeze(0)
        slots = (
            count.to(dtype=torch.long) + offsets
        ).div(4, rounding_mode="floor").clamp_max(6)
        selector = F.one_hot(slots, num_classes=7).to(encoded.dtype).unsqueeze(-1)
        candidates = selector * encoded.unsqueeze(2)
        return (
            sums + candidates.sum(dim=1),
            torch.maximum(maxima, candidates.amax(dim=1)),
            count + frames.shape[1],
        )

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        sums, maxima, _count = state
        features = torch.cat((0.25 * sums, maxima), dim=-1).flatten(1)
        return self.classifier(self.memory_norm(features))

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 3:
            return list(range(available_frames))
        return list(range(4, available_frames))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return KeywordGRU()
=======
def build_model() -> nn.Module:
    return SegmentMemoryNet()
>>>>>>> REPLACE