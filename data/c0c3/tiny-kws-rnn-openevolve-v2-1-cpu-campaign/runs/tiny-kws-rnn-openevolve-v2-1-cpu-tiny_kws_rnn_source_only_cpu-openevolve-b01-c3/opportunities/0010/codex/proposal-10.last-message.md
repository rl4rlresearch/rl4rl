MECHANISM: Four-frame hierarchical dual-rate GRU

HYPOTHESIS: A 64-unit fast GRU encoding all 16 scheduled frames into four local chunks, followed by a 112-unit slow GRU operating only at chunk boundaries, will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 28% versus the qualified 110-unit monolithic GRU.

INTENDED_EDIT: Replace the assumption that every frame needs the same full-width recurrent update with a two-timescale hierarchy: a resettable fast GRU processes each four-frame chunk, a slow GRU models the resulting four-chunk command sequence, and prediction combines the slow state with the mean fast representation.

EVIDENCE: Reducing temporal coverage from 16 to 15 frames collapsed accuracy to 80.25%, while the current 16-frame model reaches 86.01%; this suggests preserving all observations but challenges the unsupported assumption that all 16 require an expensive 110-unit transition. The proposed hierarchy keeps complete coverage and is expected to reduce recurrent MACs from 559,416,000 to about 403,092,480.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
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
        return list(range(0, available_frames, 2))
=======
class KeywordGRU(nn.Module):
    """A causal two-timescale GRU hierarchy over four-frame chunks."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_gru = nn.GRUCell(20, 64)
        self.slow_gru = nn.GRUCell(64, 112)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return fast, slow, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, summary, count = state
        fast = self.fast_gru(self.input_norm(frame), fast)
        summary = summary + fast
        count = count + 1.0

        if int(count[0, 0].detach().item()) % 4 == 0:
            slow = self.slow_gru(fast, slow)
            fast = torch.zeros_like(fast)

        return fast, slow, summary, count

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        _fast, slow, summary, count = state
        pooled_fast = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((slow, pooled_fast), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE