MECHANISM: Mixed pair–triplet causal transition packing

HYPOTHESIS: Packing the proven 26-frame window into eight three-frame transitions and one central two-frame transition with a 76-unit GRU will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 229,611,580.

INTENDED_EDIT: Replace 13 paired transitions with nine mixed-width temporal chunks, widen the GRU from 67 to 76 units, and retain three temporal summaries, maximum pooling, final-state pooling, and seven relative logits.

EVIDENCE: The 67-unit paired model reached 85.52%, showing every retained frame need not receive an individual transition, while frame deletion failed. The earlier triplet implementation was not verified and therefore provides no negative accuracy evidence; this version preserves exactly the successful 26 input frames with explicit step/sequence-equivalent phase logic.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A causal GRU that performs one learned transition per adjacent frame pair."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 67, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(335, 7)
=======
class KeywordGRU(nn.Module):
    """A causal GRU that packs 26 frames into nine learned transitions."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(60, 76, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(380, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 67, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        middle_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 67, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        middle_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 40, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        frame = self.input_norm(frame)

        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
                maximum,
                count,
                frame,
                torch.ones_like(phase),
            )

        paired = torch.cat((pending, frame), dim=1)
        output, hidden = self.gru(
            paired.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        early_weight = (count < 3.0).to(dtype=output.dtype)
        middle_weight = (
            (count >= 3.0) & (count < 6.0)
        ).to(dtype=output.dtype)
        late_weight = (count >= 6.0).to(dtype=output.dtype)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            middle_summary + middle_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        frame = self.input_norm(frame)
        phase_value = int(phase[0, 0].item())
        transition_index = int(count[0, 0].item())

        if phase_value == 0:
            buffered = torch.cat((frame, torch.zeros_like(frame)), dim=1)
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
                maximum,
                count,
                buffered,
                torch.ones_like(phase),
            )

        if phase_value == 1 and transition_index != 4:
            buffered = torch.cat((pending[:, :20], frame), dim=1)
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
                maximum,
                count,
                buffered,
                torch.full_like(phase, 2.0),
            )

        if phase_value == 1:
            midpoint = 0.5 * (pending[:, :20] + frame)
            packed = torch.cat((pending[:, :20], frame, midpoint), dim=1)
        else:
            packed = torch.cat((pending, frame), dim=1)

        output, hidden = self.gru(
            packed.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        early_weight = (count < 3.0).to(dtype=output.dtype)
        middle_weight = (
            (count >= 3.0) & (count < 6.0)
        ).to(dtype=output.dtype)
        late_weight = (count >= 6.0).to(dtype=output.dtype)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            early_summary + early_weight * output,
            middle_summary + middle_weight * output,
            late_summary + late_weight * output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        if frames.shape[1] == 0:
            return state

        frames = self.input_norm(frames)
        paired_parts: list[torch.Tensor] = []
        position = 0

        if phase[0, 0].item() >= 0.5:
            paired_parts.append(
                torch.cat((pending.unsqueeze(1), frames[:, :1, :]), dim=2)
            )
            position = 1

        pair_count = (frames.shape[1] - position) // 2
        if pair_count > 0:
            end = position + 2 * pair_count
            paired_parts.append(
                torch.cat(
                    (
                        frames[:, position:end:2, :],
                        frames[:, position + 1:end:2, :],
                    ),
                    dim=2,
                )
            )
            position = end

        if paired_parts:
            paired = torch.cat(paired_parts, dim=1)
            outputs, hidden = self.gru(
                paired,
                hidden.transpose(0, 1).contiguous(),
            )
            sequence_maximum = outputs.amax(dim=1)
            maximum = torch.where(
                count > 0,
                torch.maximum(maximum, sequence_maximum),
                sequence_maximum,
            )
            positions = count.unsqueeze(1) + torch.arange(
                paired.shape[1],
                device=outputs.device,
                dtype=outputs.dtype,
            ).view(1, -1, 1)
            early_weights = (positions < 3.0).to(dtype=outputs.dtype)
            middle_weights = (
                (positions >= 3.0) & (positions < 6.0)
            ).to(dtype=outputs.dtype)
            late_weights = (positions >= 6.0).to(dtype=outputs.dtype)
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            middle_summary = middle_summary + (
                outputs * middle_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
            count = count + paired.shape[1]
            hidden = hidden.transpose(0, 1)
            pending = torch.zeros_like(pending)
            phase = torch.zeros_like(phase)

        if position < frames.shape[1]:
            pending = frames[:, position, :]
            phase = torch.ones_like(phase)

        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        ) = state
        if frames.shape[1] == 0:
            return state

        frames = self.input_norm(frames)
        packed_parts: list[torch.Tensor] = []
        current_pending = pending
        phase_value = int(phase[0, 0].item())
        transition_index = int(count[0, 0].item())

        for position in range(frames.shape[1]):
            frame = frames[:, position, :]
            if phase_value == 0:
                current_pending = torch.cat(
                    (frame, torch.zeros_like(frame)),
                    dim=1,
                )
                phase_value = 1
            elif phase_value == 1 and transition_index != 4:
                current_pending = torch.cat(
                    (current_pending[:, :20], frame),
                    dim=1,
                )
                phase_value = 2
            elif phase_value == 1:
                midpoint = 0.5 * (current_pending[:, :20] + frame)
                packed_parts.append(
                    torch.cat(
                        (current_pending[:, :20], frame, midpoint),
                        dim=1,
                    ).unsqueeze(1)
                )
                current_pending = torch.zeros_like(current_pending)
                phase_value = 0
                transition_index += 1
            else:
                packed_parts.append(
                    torch.cat((current_pending, frame), dim=1).unsqueeze(1)
                )
                current_pending = torch.zeros_like(current_pending)
                phase_value = 0
                transition_index += 1

        if packed_parts:
            packed = torch.cat(packed_parts, dim=1)
            outputs, hidden = self.gru(
                packed,
                hidden.transpose(0, 1).contiguous(),
            )
            sequence_maximum = outputs.amax(dim=1)
            maximum = torch.where(
                count > 0,
                torch.maximum(maximum, sequence_maximum),
                sequence_maximum,
            )
            positions = count.unsqueeze(1) + torch.arange(
                packed.shape[1],
                device=outputs.device,
                dtype=outputs.dtype,
            ).view(1, -1, 1)
            early_weights = (positions < 3.0).to(dtype=outputs.dtype)
            middle_weights = (
                (positions >= 3.0) & (positions < 6.0)
            ).to(dtype=outputs.dtype)
            late_weights = (positions >= 6.0).to(dtype=outputs.dtype)
            early_summary = early_summary + (
                outputs * early_weights
            ).sum(dim=1)
            middle_summary = middle_summary + (
                outputs * middle_weights
            ).sum(dim=1)
            late_summary = late_summary + (
                outputs * late_weights
            ).sum(dim=1)
            count = count + packed.shape[1]
            hidden = hidden.transpose(0, 1)

        pending = current_pending
        phase = torch.full_like(phase, float(phase_value))
        return (
            hidden,
            early_summary,
            middle_summary,
            late_summary,
            maximum,
            count,
            pending,
            phase,
        )
>>>>>>> REPLACE