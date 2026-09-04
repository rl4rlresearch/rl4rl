MECHANISM: Lossless causal frame-pair transitions

HYPOTHESIS: A 70-unit GRU that jointly consumes adjacent frames from the verified 26-frame window will retain at least 85% accuracy while reducing learned GRU transitions from 25 to 13 and estimated total validation MACs from 277.66M to 246.11M.

INTENDED_EDIT: Cache the first frame of each causal pair without a dense operation, concatenate it with the next frame, and execute one wider GRU transition per pair; vectorize the identical paired computation during sequence training.

EVIDENCE: The 58-unit model reached 86.01% on frames 3–28, while deleting frames and compensating with width failed. This challenges the load-bearing assumption that every retained frame needs its own full recurrent transition: all 26 proven-useful frames remain available, but temporal updates occur at a learned two-frame clock.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A compact causal GRU with mean, maximum, and final temporal features."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 58, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 58, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return hidden.transpose(0, 1), summary + output, maximum, count + 1.0

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        sequence_maximum = outputs.amax(dim=1)
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, sequence_maximum),
            sequence_maximum,
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            maximum,
            count + frames.shape[1],
        )

    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(25, available_frames)
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))
=======
class KeywordGRU(nn.Module):
    """A causal GRU that performs one learned transition per adjacent frame pair."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(40, 70, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(210, 8)

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
        hidden = torch.zeros(batch_size, 1, 70, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 70, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pending = torch.zeros(batch_size, 20, device=device, dtype=dtype)
        phase = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, maximum, count, pending, phase

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
        hidden, summary, maximum, count, pending, phase = state
        frame = self.input_norm(frame)

        if phase[0, 0].item() < 0.5:
            return (
                hidden,
                summary,
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
        maximum = torch.where(
            count > 0,
            torch.maximum(maximum, output),
            output,
        )
        return (
            hidden.transpose(0, 1),
            summary + output,
            maximum,
            count + 1.0,
            torch.zeros_like(pending),
            torch.zeros_like(phase),
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
        hidden, summary, maximum, count, pending, phase = state
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
            summary = summary + outputs.sum(dim=1)
            count = count + paired.shape[1]
            hidden = hidden.transpose(0, 1)
            pending = torch.zeros_like(pending)
            phase = torch.zeros_like(phase)

        if position < frames.shape[1]:
            pending = frames[:, position, :]
            phase = torch.ones_like(phase)

        return hidden, summary, maximum, count, pending, phase

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
        hidden, summary, maximum, count, pending, phase = state
        del pending, phase
        pooled = torch.cat(
            (summary / count.clamp_min(1.0), maximum, hidden[:, 0, :]),
            dim=1,
        )
        return self.classifier(pooled)

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(26, available_frames)
        steps -= steps % 2
        start = (available_frames - steps) // 2
        return list(range(start, start + steps))
>>>>>>> REPLACE