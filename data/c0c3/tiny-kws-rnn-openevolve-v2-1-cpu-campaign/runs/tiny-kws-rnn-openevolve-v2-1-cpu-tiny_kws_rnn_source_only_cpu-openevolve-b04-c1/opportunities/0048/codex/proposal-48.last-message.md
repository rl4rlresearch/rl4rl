MECHANISM: Causal three-frame transition packing

HYPOTHESIS: A 76-unit GRU consuming 27 central frames in nine causal three-frame transitions will retain at least 85% validation accuracy while reducing estimated total inference MACs from 229,777,840 to 229,611,580.

INTENDED_EDIT: Cache two frames, concatenate each causal frame triplet into a 60-feature GRU input, widen the hidden state to 76 units, and use three equal transition summaries plus maximum and final-state views.

EVIDENCE: Pairing preserved all useful frames and reached 85.77% with only 13 learned transitions, whereas deleting frames failed; triplet packing tests whether further transition reduction can preserve accuracy while a 76-unit width keeps recurrent MACs slightly below the successful 67-unit paired model.

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
    """A causal GRU that performs one learned transition per three-frame group."""

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
=======
        hidden = torch.zeros(batch_size, 1, 76, device=device, dtype=dtype)
        early_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        middle_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        late_summary = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 76, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 2, 20, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        if phase[0, 0].item() < 0.5:
            pending = torch.stack((frame, torch.zeros_like(frame)), dim=1)
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
                maximum,
                count,
                pending,
                torch.ones_like(phase),
            )

        if phase[0, 0].item() < 1.5:
            pending = torch.stack((pending[:, 0, :], frame), dim=1)
            return (
                hidden,
                early_summary,
                middle_summary,
                late_summary,
                maximum,
                count,
                pending,
                torch.full_like(phase, 2.0),
            )

        grouped = torch.cat(
            (pending[:, 0, :], pending[:, 1, :], frame),
            dim=1,
        )
        output, hidden = self.gru(
            grouped.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        grouped_parts: list[torch.Tensor] = []
        position = 0

        if phase[0, 0].item() >= 1.5:
            grouped_parts.append(
                torch.cat((pending, frames[:, :1, :]), dim=2)
            )
            position = 1
        elif phase[0, 0].item() >= 0.5:
            if frames.shape[1] == 1:
                pending = torch.stack(
                    (pending[:, 0, :], frames[:, 0, :]),
                    dim=1,
                )
                return (
                    hidden,
                    early_summary,
                    middle_summary,
                    late_summary,
                    maximum,
                    count,
                    pending,
                    torch.full_like(phase, 2.0),
                )
            grouped_parts.append(
                torch.cat((pending[:, :1, :], frames[:, :2, :]), dim=2)
            )
            position = 2

        group_count = (frames.shape[1] - position) // 3
        if group_count > 0:
            end = position + 3 * group_count
            grouped_parts.append(
                torch.cat(
                    (
                        frames[:, position:end:3, :],
                        frames[:, position + 1:end:3, :],
                        frames[:, position + 2:end:3, :],
                    ),
                    dim=2,
                )
            )
            position = end

        if grouped_parts:
            grouped = torch.cat(grouped_parts, dim=1)
            outputs, hidden = self.gru(
                grouped,
                hidden.transpose(0, 1).contiguous(),
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                paired.shape[1],
                device=outputs.device,
                dtype=outputs.dtype,
            ).view(1, -1, 1)
=======
                grouped.shape[1],
                device=outputs.device,
                dtype=outputs.dtype,
            ).view(1, -1, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
            count = count + paired.shape[1]
            hidden = hidden.transpose(0, 1)
            pending = torch.zeros_like(pending)
            phase = torch.zeros_like(phase)

        if position < frames.shape[1]:
            pending = frames[:, position, :]
            phase = torch.ones_like(phase)
=======
            count = count + grouped.shape[1]
            hidden = hidden.transpose(0, 1)
            pending = torch.zeros_like(pending)
            phase = torch.zeros_like(phase)

        remaining = frames.shape[1] - position
        if remaining == 1:
            last_frame = frames[:, position, :]
            pending = torch.stack(
                (last_frame, torch.zeros_like(last_frame)),
                dim=1,
            )
            phase = torch.ones_like(phase)
        elif remaining == 2:
            pending = frames[:, position : position + 2, :]
            phase = torch.full_like(phase, 2.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(26, available_frames)
        steps -= steps % 2
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(27, available_frames)
        steps -= steps % 3
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))
>>>>>>> REPLACE