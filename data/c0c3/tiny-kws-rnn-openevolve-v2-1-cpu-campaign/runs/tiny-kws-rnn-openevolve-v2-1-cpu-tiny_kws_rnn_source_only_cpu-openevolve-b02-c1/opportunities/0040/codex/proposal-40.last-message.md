MECHANISM: Hierarchical two-rate gated memory

HYPOTHESIS: A 48-unit GRU updated every frame plus a 64-unit GRU updated after each frame pair will retain at least 85% validation accuracy while reducing expected total inference MACs from 551,393,140 to approximately 454,743,920.

INTENDED_EDIT: Replace the single same-rate 79-unit GRU with fully gated fast and slow GRU cells; the fast state integrates all 25 frames, while the slow state receives contextualized fast states only at 12 pair boundaries. Predict from mean-fast, final-fast, and slow memories.

EVIDENCE: Full GRU gating is load-bearing—the unified 79-unit GRU reached 86.75%, whereas coupled-gate alternatives missed 85%. This challenges the shared assumption that all memory capacity must execute at every frame while preserving full GRU gates, the verified input features, temporal coverage, and comparable learned capacity.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """Unified gated memory over spectral levels and explicit local dynamics."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(35, 79, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(158, 7)
=======
class KeywordGRU(nn.Module):
    """Hierarchical gated memory with frame-rate and pair-rate state updates."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_cell = nn.GRUCell(35, 48)
        self.slow_cell = nn.GRUCell(48, 64)
        self.classifier = nn.Linear(160, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 79, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 79, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, previous, summary, count
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
        fast = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        slow = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        previous = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return fast, slow, previous, summary, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        delta_features = torch.cat(
            (
                delta[:, :10],
                delta[:, 10:12].mean(dim=1, keepdim=True),
                delta[:, 12:14].mean(dim=1, keepdim=True),
                delta[:, 14:16].mean(dim=1, keepdim=True),
                delta[:, 16:18].mean(dim=1, keepdim=True),
                delta[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        features = torch.cat((normalized, delta_features), dim=1).unsqueeze(1)
        output, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        current = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            normalized,
            summary + current,
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
        fast, slow, previous, summary, count = state
        normalized = self.input_norm(frame)
        has_previous = (count > 0.0).to(dtype=normalized.dtype)
        delta = (normalized - previous) * has_previous
        delta_features = torch.cat(
            (
                delta[:, :10],
                delta[:, 10:12].mean(dim=1, keepdim=True),
                delta[:, 12:14].mean(dim=1, keepdim=True),
                delta[:, 14:16].mean(dim=1, keepdim=True),
                delta[:, 16:18].mean(dim=1, keepdim=True),
                delta[:, 18:].mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        features = torch.cat((normalized, delta_features), dim=1)
        fast = self.fast_cell(features, fast)

        pair_boundary = torch.remainder(
            count[:, 0].to(dtype=torch.long), 2
        ) == 1
        if bool(pair_boundary.any()):
            updated = self.slow_cell(
                fast[pair_boundary], slow[pair_boundary]
            )
            next_slow = slow.clone()
            next_slow[pair_boundary] = updated
            slow = next_slow

        return fast, slow, normalized, summary + fast, count + 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, previous, summary, count = state
        normalized = self.input_norm(frames)
        has_previous = (count > 0.0).to(dtype=normalized.dtype).unsqueeze(1)
        first_delta = (
            normalized[:, :1, :] - previous.unsqueeze(1)
        ) * has_previous
        remaining_deltas = normalized[:, 1:, :] - normalized[:, :-1, :]
        deltas = torch.cat((first_delta, remaining_deltas), dim=1)
        delta_features = torch.cat(
            (
                deltas[:, :, :10],
                deltas[:, :, 10:12].mean(dim=2, keepdim=True),
                deltas[:, :, 12:14].mean(dim=2, keepdim=True),
                deltas[:, :, 14:16].mean(dim=2, keepdim=True),
                deltas[:, :, 16:18].mean(dim=2, keepdim=True),
                deltas[:, :, 18:].mean(dim=2, keepdim=True),
            ),
            dim=2,
        )
        features = torch.cat((normalized, delta_features), dim=2)
        outputs, hidden = self.gru(
            features, hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            normalized[:, -1, :],
            summary + outputs.sum(dim=1),
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
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, _previous, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        readout = torch.cat((mean_output, hidden[:, 0, :]), dim=1)
        coordinates = self.classifier(readout)
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
        fast, slow, _previous, summary, count = state
        mean_fast = summary / count.clamp_min(1.0)
        readout = torch.cat((mean_fast, fast, slow), dim=1)
        coordinates = self.classifier(readout)
>>>>>>> REPLACE