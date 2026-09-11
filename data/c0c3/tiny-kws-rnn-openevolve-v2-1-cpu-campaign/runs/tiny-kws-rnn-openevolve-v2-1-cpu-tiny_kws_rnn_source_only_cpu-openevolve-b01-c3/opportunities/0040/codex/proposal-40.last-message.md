MECHANISM: Ordered two-frame block recurrence with mean-max-endpoint readout

HYPOTHESIS: A 128-unit gated model that retains all 16 sampled frames but performs one learned transition per ordered two-frame block will reach at least 85% validation accuracy while reducing total inference MACs to approximately 262,886,400.

INTENDED_EDIT: Replace per-frame learned transitions with alternating frame buffering and eight gated block transitions over local mean/difference features; widen the state to 128 units and classify from temporal mean, maximum, and final state.

EVIDENCE: The 100-unit 31/32/33 model reached 85.890% accuracy, showing the candidate-biased gated mechanism has useful margin. Unlike the failed 15-frame reduction, this patch preserves all 16 observations and challenges the load-bearing assumption that every observation requires a separate expensive recurrent transition.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A gated recurrent memory with selectively compressed feedback."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 312)
        self.recurrent_down = nn.Linear(104, 95)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 104),
                nn.Linear(32, 104),
                nn.Linear(32, 104),
            ]
        )
        self.classifier = nn.Linear(208, 8)
=======
class KeywordGRU(nn.Module):
    """A gated memory that consumes ordered pairs in learned block updates."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(40, 384)
        self.recurrent_down = nn.Linear(128, 96)
        self.recurrent_up = nn.ModuleList(
            [
                nn.Linear(31, 128),
                nn.Linear(32, 128),
                nn.Linear(33, 128),
            ]
        )
        self.classifier = nn.Linear(384, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count = state
        hidden = hidden[:, 0, :]

        input_reset, input_update, input_candidate = self.input_projection(
            self.input_norm(frame)
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 32), dim=1
        )

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return hidden.unsqueeze(1), summary + hidden, count + 1.0
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        maximum = torch.full(
            (batch_size, 128), -1.0, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        buffered = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count, buffered, phase

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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden, summary, maximum, count, buffered, phase = state
        normalized = self.input_norm(frame)

        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                summary,
                maximum,
                count,
                normalized,
                torch.ones_like(phase),
            )

        hidden = hidden[:, 0, :]
        pair_features = torch.cat(
            (
                0.5 * (buffered + normalized),
                normalized - buffered,
            ),
            dim=1,
        )
        input_reset, input_update, input_candidate = self.input_projection(
            pair_features
        ).chunk(3, dim=1)
        reset_code, update_code, candidate_code = torch.split(
            torch.tanh(self.recurrent_down(hidden)), (31, 32, 33), dim=1
        )

        reset = torch.sigmoid(
            input_reset + self.recurrent_up[0](reset_code)
        )
        update = torch.sigmoid(
            input_update + self.recurrent_up[1](update_code)
        )
        candidate = torch.tanh(
            input_candidate
            + reset * self.recurrent_up[2](candidate_code)
        )
        hidden = update * hidden + (1.0 - update) * candidate

        return (
            hidden.unsqueeze(1),
            summary + hidden,
            torch.maximum(maximum, hidden),
            count + 1.0,
            torch.zeros_like(buffered),
            torch.zeros_like(phase),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=1
        )
        return self.classifier(features)
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
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state

    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count, buffered, phase = state
        del buffered, phase
        features = torch.cat(
            (
                summary / count.clamp_min(1.0),
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
        return self.classifier(features)
>>>>>>> REPLACE