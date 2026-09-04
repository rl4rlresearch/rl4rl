MECHANISM: Paired-frame multirate recurrent pyramid

HYPOTHESIS: A 112-unit full-rate GRU augmented by a 40-unit GRU that processes ten adjacent-frame averages will retain at least 85% accuracy while reducing total validation inference MACs from 823,084,800 to approximately 783,599,680.

INTENDED_EDIT: Replace the monolithic 122-unit recurrence with fine- and coarse-timescale recurrent states, then fuse their mean and endpoint representations for prediction.

EVIDENCE: The 120-unit, 20-frame dual-readout model reached 86.26% at 823,084,800 MACs. This tests the load-bearing assumption that all recurrent capacity must execute on every frame: most capacity remains full-rate, while a structurally cheaper branch receives all frames through pairwise temporal pooling.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A causal GRU with complementary mean and endpoint readouts."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 122, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(122, 8)
        self.endpoint_classifier = nn.Linear(122, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)
=======
class KeywordGRU(nn.Module):
    """A causal multirate GRU with fine and pair-pooled temporal states."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.fast_gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.slow_gru = nn.GRU(20, 40, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(152, 8)
        self.endpoint_classifier = nn.Linear(152, 8)
        nn.init.zeros_(self.endpoint_classifier.weight)
        nn.init.zeros_(self.endpoint_classifier.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 122, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 122, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
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
        fast_hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        slow_hidden = torch.zeros(batch_size, 1, 40, device=device, dtype=dtype)
        fast_summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
        slow_summary = torch.zeros(batch_size, 40, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            pending,
            count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            pending,
            count,
        ) = state
        normalized = self.input_norm(frame)
        fast_output, fast_hidden = self.fast_gru(
            normalized.unsqueeze(1),
            fast_hidden.transpose(0, 1).contiguous(),
        )
        fast_summary = fast_summary + fast_output[:, 0, :]

        if int(count[0, 0].item()) % 2 == 1:
            paired = 0.5 * (pending + normalized)
            slow_output, slow_hidden = self.slow_gru(
                paired.unsqueeze(1),
                slow_hidden.transpose(0, 1).contiguous(),
            )
            slow_summary = slow_summary + slow_output[:, 0, :]

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary,
            slow_summary,
            normalized,
            count + 1.0,
        )

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
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            pending,
            count,
        ) = state
        normalized = self.input_norm(frames)
        fast_outputs, fast_hidden = self.fast_gru(
            normalized,
            fast_hidden.transpose(0, 1).contiguous(),
        )
        fast_summary = fast_summary + fast_outputs.sum(dim=1)

        if int(count[0, 0].item()) % 2 == 0:
            pair_count = frames.shape[1] // 2
            paired = 0.5 * (
                normalized[:, : 2 * pair_count : 2, :]
                + normalized[:, 1 : 2 * pair_count : 2, :]
            )
        else:
            pair_parts = [
                0.5 * (pending.unsqueeze(1) + normalized[:, :1, :])
            ]
            remaining_pairs = (frames.shape[1] - 1) // 2
            if remaining_pairs > 0:
                pair_parts.append(
                    0.5
                    * (
                        normalized[:, 1 : 1 + 2 * remaining_pairs : 2, :]
                        + normalized[:, 2 : 1 + 2 * remaining_pairs : 2, :]
                    )
                )
            paired = torch.cat(pair_parts, dim=1)

        if paired.shape[1] > 0:
            slow_outputs, slow_hidden = self.slow_gru(
                paired,
                slow_hidden.transpose(0, 1).contiguous(),
            )
            slow_summary = slow_summary + slow_outputs.sum(dim=1)

        return (
            fast_hidden.transpose(0, 1),
            slow_hidden.transpose(0, 1),
            fast_summary,
            slow_summary,
            normalized[:, -1, :],
            count + frames.shape[1],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        endpoint = hidden[:, 0, :]
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)
=======
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
        (
            fast_hidden,
            slow_hidden,
            fast_summary,
            slow_summary,
            _pending,
            count,
        ) = state
        slow_count = torch.floor(0.5 * count).clamp_min(1.0)
        mean_output = torch.cat(
            (
                fast_summary / count.clamp_min(1.0),
                slow_summary / slow_count,
            ),
            dim=1,
        )
        endpoint = torch.cat(
            (fast_hidden[:, 0, :], slow_hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(mean_output) + self.endpoint_classifier(endpoint)
>>>>>>> REPLACE