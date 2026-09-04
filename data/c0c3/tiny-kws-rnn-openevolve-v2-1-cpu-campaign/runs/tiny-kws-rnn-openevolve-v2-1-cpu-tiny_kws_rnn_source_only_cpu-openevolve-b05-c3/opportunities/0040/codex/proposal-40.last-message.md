MECHANISM: Clocked hierarchical dual-timescale GRU

HYPOTHESIS: An 80-unit fast GRU updated for all 20 scheduled frames plus an 80-unit slow GRU updated from four-frame summaries will retain at least 85% validation accuracy while reducing expected total inference MACs from 643,181,700 to approximately 549,244,800.

INTENDED_EDIT: Replace the monolithic recurrent state with fast and slow GRUCell states, execute the slow transition every fourth frame, classify from fast-scale and slow-scale summaries, and use the proven batch-64 training regime.

EVIDENCE: The 105-unit monolithic GRU achieved 86.75% at 643,181,700 MACs, while the custom bottleneck-gated recurrence narrowly missed at 84.91%. This challenges the shared assumption that one full-width transition must process every frame while retaining standard, well-qualified GRU gating at both timescales.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
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
        return list(range(available_frames))
=======
class KeywordGRU(nn.Module):
    """A clocked hierarchy with separate acoustic and command timescales."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_cell = nn.GRUCell(20, 80)
        self.slow_norm = nn.LayerNorm(80)
        self.slow_cell = nn.GRUCell(80, 80)
        self.classifier = nn.Linear(240, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        fast = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 80, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        slow_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            fast,
            slow,
            block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        )

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
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
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            fast,
            slow,
            block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        ) = state

        fast = self.fast_cell(self.input_norm(frame), fast)
        block_sum = block_sum + fast
        fast_summary = fast_summary + fast
        count = count + 1.0

        if int(count[0, 0].detach().item()) % 4 == 0:
            slow_input = self.slow_norm(block_sum * 0.25)
            slow = self.slow_cell(slow_input, slow)
            slow_summary = slow_summary + slow
            slow_count = slow_count + 1.0
            block_sum = torch.zeros_like(block_sum)

        return (
            fast,
            slow,
            block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        )

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            _fast,
            slow,
            _block_sum,
            fast_summary,
            slow_summary,
            count,
            slow_count,
        ) = state
        fast_mean = fast_summary / count.clamp_min(1.0)
        slow_mean = slow_summary / slow_count.clamp_min(1.0)
        return self.classifier(torch.cat((fast_mean, slow_mean, slow), dim=-1))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
>>>>>>> REPLACE