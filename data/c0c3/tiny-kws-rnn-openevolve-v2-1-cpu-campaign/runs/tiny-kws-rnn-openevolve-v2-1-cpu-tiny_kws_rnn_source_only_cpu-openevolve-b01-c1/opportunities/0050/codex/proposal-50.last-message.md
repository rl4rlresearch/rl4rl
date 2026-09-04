MECHANISM: Lossless pair-concatenated pyramidal upper recurrence

HYPOTHESIS: Updating the upper GRU 14 times from concatenated adjacent 58-feature inputs will retain at least 85% accuracy while saving approximately 78.9M recurrent MACs, because neither frame’s lower representation or acoustic skip is discarded.

INTENDED_EDIT: Widen the upper GRU input to 116, buffer alternating upper inputs, execute one upper update per pair, track its count separately, and make sequence execution equivalent through vectorized temporal pairing.

EVIDENCE: The prior half-rate model narrowly missed at 84.79% after averaging acoustics and discarding every first lower output, while the current model reaches 86.50%. This challenges the assumption that upper recurrence must run every frame while directly addressing the failed design’s temporal information loss.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A raw-skip hierarchy of two compact causal recurrent stages."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(19)
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_gru = nn.GRU(58, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
=======
class KeywordGRU(nn.Module):
    """A full-rate lower GRU with lossless pairwise pyramidal upper recurrence."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(19)
        self.lower_gru = nn.GRU(19, 49, num_layers=1, batch_first=True)
        self.upper_band45 = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.upper_band45.weight, 0.5)
        self.upper_gru = nn.GRU(116, 48, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(194, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return lower_hidden, upper_hidden, lower_summary, upper_summary, count
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
        torch.Tensor,
    ]:
        lower_hidden = torch.zeros(batch_size, 1, 49, device=device, dtype=dtype)
        upper_hidden = torch.zeros(batch_size, 1, 48, device=device, dtype=dtype)
        pending_upper_input = torch.zeros(
            batch_size, 58, device=device, dtype=dtype
        )
        lower_summary = torch.zeros(batch_size, 49, device=device, dtype=dtype)
        upper_summary = torch.zeros(batch_size, 48, device=device, dtype=dtype)
        frame_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        upper_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            lower_hidden,
            upper_hidden,
            pending_upper_input,
            lower_summary,
            upper_summary,
            frame_count,
            upper_count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frame))
        lower_output, lower_hidden = self.lower_gru(
            normalized.unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_input = torch.cat((upper_acoustic, lower_output), dim=-1)
        upper_output, upper_hidden = self.upper_gru(
            upper_input.unsqueeze(1),
            upper_hidden.transpose(0, 1).contiguous(),
        )
        upper_output = upper_output[:, 0, :]
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_output,
            upper_summary + upper_output,
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
            lower_hidden,
            upper_hidden,
            pending_upper_input,
            lower_summary,
            upper_summary,
            frame_count,
            upper_count,
        ) = state
        normalized = self.input_norm(self._fold_bands(frame))
        lower_output, lower_hidden = self.lower_gru(
            normalized.unsqueeze(1),
            lower_hidden.transpose(0, 1).contiguous(),
        )
        lower_output = lower_output[:, 0, :]
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_frame_input = torch.cat((upper_acoustic, lower_output), dim=-1)

        if int(frame_count[0, 0].item()) % 2 == 1:
            paired_input = torch.cat(
                (pending_upper_input, upper_frame_input), dim=-1
            )
            upper_output, upper_hidden = self.upper_gru(
                paired_input.unsqueeze(1),
                upper_hidden.transpose(0, 1).contiguous(),
            )
            upper_output = upper_output[:, 0, :]
            pending_upper_input = torch.zeros_like(pending_upper_input)
            upper_summary = upper_summary + upper_output
            upper_count = upper_count + 1.0
        else:
            pending_upper_input = upper_frame_input

        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            pending_upper_input,
            lower_summary + lower_output,
            upper_summary,
            frame_count + 1.0,
            upper_count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        normalized = self.input_norm(self._fold_bands(frames))
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_inputs = torch.cat((upper_acoustic, lower_outputs), dim=-1)
        upper_outputs, upper_hidden = self.upper_gru(
            upper_inputs,
            upper_hidden.transpose(0, 1).contiguous(),
        )
        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            lower_summary + lower_outputs.sum(dim=1),
            upper_summary + upper_outputs.sum(dim=1),
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
        if frames.shape[1] == 0:
            return state

        (
            lower_hidden,
            upper_hidden,
            pending_upper_input,
            lower_summary,
            upper_summary,
            frame_count,
            upper_count,
        ) = state
        normalized = self.input_norm(self._fold_bands(frames))
        lower_outputs, lower_hidden = self.lower_gru(
            normalized,
            lower_hidden.transpose(0, 1).contiguous(),
        )
        upper_acoustic = torch.cat(
            (
                normalized[..., :4],
                self.upper_band45(normalized[..., 4:6]),
                normalized[..., 6:8].mean(dim=-1, keepdim=True),
                normalized[..., 8:10].mean(dim=-1, keepdim=True),
                normalized[..., 10:12].mean(dim=-1, keepdim=True),
                normalized[..., 12:14].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        upper_frame_inputs = torch.cat(
            (upper_acoustic, lower_outputs), dim=-1
        )

        pair_chunks: list[torch.Tensor] = []
        offset = 0
        if int(frame_count[0, 0].item()) % 2 == 1:
            pair_chunks.append(
                torch.cat(
                    (
                        pending_upper_input.unsqueeze(1),
                        upper_frame_inputs[:, :1, :],
                    ),
                    dim=-1,
                )
            )
            offset = 1

        remaining = upper_frame_inputs[:, offset:, :]
        pairable = (remaining.shape[1] // 2) * 2
        if pairable:
            pair_chunks.append(
                remaining[:, :pairable, :].reshape(
                    remaining.shape[0], pairable // 2, 116
                )
            )

        if remaining.shape[1] > pairable:
            pending_upper_input = remaining[:, -1, :]
        else:
            pending_upper_input = torch.zeros_like(pending_upper_input)

        if pair_chunks:
            upper_inputs = torch.cat(pair_chunks, dim=1)
            upper_outputs, upper_hidden = self.upper_gru(
                upper_inputs,
                upper_hidden.transpose(0, 1).contiguous(),
            )
            upper_summary = upper_summary + upper_outputs.sum(dim=1)
            upper_count = upper_count + upper_inputs.shape[1]

        return (
            lower_hidden.transpose(0, 1),
            upper_hidden.transpose(0, 1),
            pending_upper_input,
            lower_summary + lower_outputs.sum(dim=1),
            upper_summary,
            frame_count + frames.shape[1],
            upper_count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        lower_hidden, upper_hidden, lower_summary, upper_summary, count = state
        divisor = count.clamp_min(1.0)
        features = torch.cat(
            (
                lower_summary / divisor,
                upper_summary / divisor,
                lower_hidden[:, 0, :],
                upper_hidden[:, 0, :],
            ),
            dim=-1,
        )
        return self.classifier(features)
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
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        (
            lower_hidden,
            upper_hidden,
            _pending_upper_input,
            lower_summary,
            upper_summary,
            frame_count,
            upper_count,
        ) = state
        features = torch.cat(
            (
                lower_summary / frame_count.clamp_min(1.0),
                upper_summary / upper_count.clamp_min(1.0),
                lower_hidden[:, 0, :],
                upper_hidden[:, 0, :],
            ),
            dim=-1,
        )
        return self.classifier(features)
>>>>>>> REPLACE