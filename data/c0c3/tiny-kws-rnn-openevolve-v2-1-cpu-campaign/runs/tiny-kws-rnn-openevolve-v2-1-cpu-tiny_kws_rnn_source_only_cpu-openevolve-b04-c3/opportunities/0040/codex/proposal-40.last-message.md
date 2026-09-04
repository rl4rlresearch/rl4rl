MECHANISM: Pair-pooled multirate hierarchical recurrence

HYPOTHESIS: A 48-unit acoustic GRU processing all 23 qualified frames, paired with a 120-unit contextual GRU executing only 12 pooled updates, will retain at least 85% validation accuracy while reducing total inference MACs from 845,791,515 to approximately 777,236,160.

INTENDED_EDIT: Replace synchronous monolithic recurrence with a full-rate acoustic state and a wider half-rate contextual state; aggregate adjacent acoustic outputs before contextual updates while preserving the proven mean/terminal readout and auxiliary supervision.

EVIDENCE: The 48/87 synchronous hierarchy qualified at 85.52% with 23 frames, while both 22-frame variants failed. This suggests retaining every acoustic observation is load-bearing, but does not establish that the expensive contextual transition must execute on every frame; pair pooling preserves all 23 observations and widening the half-rate context from 87 to 120 provides capacity for the coarser temporal representation.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(240, 8)
        self.mean_aux = nn.Linear(120, 8)
        self.terminal_aux = nn.Linear(120, 8)
        self._aux_logits = None
=======
class KeywordGRU(nn.Module):
    """A multirate causal hierarchy with pair-pooled contextual updates."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.acoustic_gru = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.context_norm = nn.LayerNorm(48)
        self.context_gru = nn.GRU(48, 120, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(168)
        self.terminal_norm = nn.LayerNorm(168)
        self.classifier = nn.Linear(336, 8)
        self.mean_aux = nn.Linear(168, 8)
        self.terminal_aux = nn.Linear(168, 8)
        self._aux_logits = None
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        acoustic_hidden = torch.zeros(
            batch_size, 1, 48, device=device, dtype=dtype
        )
        context_hidden = torch.zeros(
            batch_size, 1, 120, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 120, device=device, dtype=dtype
        )
        block_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        frame_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        context_count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            block_summary,
            frame_count,
            context_count,
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
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            block_summary,
            frame_count,
            context_count,
        ) = state
        acoustic_output, next_acoustic_hidden = self.acoustic_gru(
            self.input_norm(frame).unsqueeze(1),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        acoustic_output = acoustic_output[:, 0, :]
        acoustic_hidden = next_acoustic_hidden.transpose(0, 1)
        acoustic_summary = acoustic_summary + acoustic_output

        frame_index = int(frame_count[0, 0].detach().item())
        if frame_index % 2 == 0:
            if frame_index == 0:
                context_input = acoustic_output
            else:
                context_input = 0.5 * (
                    block_summary + acoustic_output
                )
            context_output, next_context_hidden = self.context_gru(
                self.context_norm(context_input).unsqueeze(1),
                context_hidden.transpose(0, 1).contiguous(),
            )
            context_hidden = next_context_hidden.transpose(0, 1)
            context_summary = context_summary + context_output[:, 0, :]
            context_count = context_count + 1.0
            block_summary = torch.zeros_like(block_summary)
        else:
            block_summary = acoustic_output

        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            block_summary,
            frame_count + 1.0,
            context_count,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        for frame_index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, frame_index, :], state)
        return state
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        mean = summary / count.clamp_min(1.0)
        terminal = hidden[:, 0, :]
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits
=======
    def classify(
        self, state: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            _block_summary,
            frame_count,
            context_count,
        ) = state
        mean = self.mean_norm(
            torch.cat(
                (
                    acoustic_summary / frame_count.clamp_min(1.0),
                    context_summary / context_count.clamp_min(1.0),
                ),
                dim=-1,
            )
        )
        terminal = self.terminal_norm(
            torch.cat(
                (
                    acoustic_hidden[:, 0, :],
                    context_hidden[:, 0, :],
                ),
                dim=-1,
            )
        )
        logits = self.classifier(torch.cat((mean, terminal), dim=-1))
        if self.training:
            self._aux_logits = (
                self.mean_aux(mean),
                self.terminal_aux(terminal),
            )
        else:
            self._aux_logits = None
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
        return schedule
=======
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE