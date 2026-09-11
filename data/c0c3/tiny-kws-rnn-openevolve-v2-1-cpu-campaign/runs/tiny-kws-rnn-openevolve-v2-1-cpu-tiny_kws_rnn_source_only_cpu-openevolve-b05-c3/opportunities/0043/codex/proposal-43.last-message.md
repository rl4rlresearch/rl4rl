MECHANISM: Adjacent-width clocked dual-timescale GRU

HYPOTHESIS: A 78-unit fast GRU over all 20 scheduled frames plus a 78-unit slow GRU over four five-frame summaries will retain at least 85% validation accuracy while reducing total inference MACs from 506,066,100 to approximately 494,320,320.

INTENDED_EDIT: Replace the 106-unit monolithic GRU with 78-unit fast and slow GRUCells, update the slow state every fifth step from the mean fast-state block, and classify from fast-mean, slow-mean, and slow-endpoint summaries.

EVIDENCE: The adjacent 79-unit dual-timescale design achieved 86.26% accuracy at 506,066,100 MACs, leaving a 1.26-point margin; reducing both recurrent widths by one unit is the most direct remaining capacity-boundary probe.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 106, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(106, 8)
        self.endpoint_classifier = nn.Linear(106, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 106, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 106, device=device, dtype=dtype)
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
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(20, available_frames)
        return [
            (index * (available_frames - 1)) // (steps - 1)
            for index in range(steps)
        ]
=======
class KeywordGRU(nn.Module):
    """A clocked hierarchy with separate acoustic and command timescales."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_cell = nn.GRUCell(20, 78)
        self.slow_norm = nn.LayerNorm(78)
        self.slow_cell = nn.GRUCell(78, 78)
        self.classifier = nn.Linear(234, 8)

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
        fast = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        block_sum = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 78, device=device, dtype=dtype)
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

        if int(count[0, 0].detach().item()) % 5 == 0:
            slow = self.slow_cell(self.slow_norm(block_sum * 0.2), slow)
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