MECHANISM: Hierarchical recurrent factorization with 25-step late-biased temporal thinning

HYPOTHESIS: The qualified 48/90 hierarchical GRU will retain at least 85% validation accuracy after removing one additional early frame, while reducing total inference MACs from 998,831,400 to approximately 960,484,020.

INTENDED_EDIT: Replace the monolithic GRU with the qualified hierarchical dual-readout model, retain auxiliary supervision, and reduce its schedule from 26 to 25 recurrent steps.

EVIDENCE: The identical 48/90 hierarchy achieved 88.22% accuracy at 26 steps—3.22 percentage points above the threshold—and previously lost only 0.12 points when reduced from 27 to 26 steps.

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
    """A hierarchical causal GRU with fine and contextual temporal state."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.acoustic_gru = nn.GRU(20, 48, num_layers=1, batch_first=True)
        self.context_norm = nn.LayerNorm(48)
        self.context_gru = nn.GRU(48, 90, num_layers=1, batch_first=True)
        self.mean_norm = nn.LayerNorm(138)
        self.terminal_norm = nn.LayerNorm(138)
        self.classifier = nn.Linear(276, 8)
        self.mean_aux = nn.Linear(138, 8)
        self.terminal_aux = nn.Linear(138, 8)
        self._aux_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        acoustic_hidden = torch.zeros(
            batch_size, 1, 48, device=device, dtype=dtype
        )
        context_hidden = torch.zeros(
            batch_size, 1, 90, device=device, dtype=dtype
        )
        acoustic_summary = torch.zeros(
            batch_size, 48, device=device, dtype=dtype
        )
        context_summary = torch.zeros(
            batch_size, 90, device=device, dtype=dtype
        )
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_output, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frame).unsqueeze(1),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_output, context_hidden = self.context_gru(
            self.context_norm(acoustic_output),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_output[:, 0, :],
            context_summary + context_output[:, 0, :],
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
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        acoustic_outputs, acoustic_hidden = self.acoustic_gru(
            self.input_norm(frames),
            acoustic_hidden.transpose(0, 1).contiguous(),
        )
        context_outputs, context_hidden = self.context_gru(
            self.context_norm(acoustic_outputs),
            context_hidden.transpose(0, 1).contiguous(),
        )
        return (
            acoustic_hidden.transpose(0, 1),
            context_hidden.transpose(0, 1),
            acoustic_summary + acoustic_outputs.sum(dim=1),
            context_summary + context_outputs.sum(dim=1),
            count + frames.shape[1],
        )

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
        (
            acoustic_hidden,
            context_hidden,
            acoustic_summary,
            context_summary,
            count,
        ) = state
        divisor = count.clamp_min(1.0)
        mean = self.mean_norm(
            torch.cat(
                (
                    acoustic_summary / divisor,
                    context_summary / divisor,
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

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
            schedule.pop(0)
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del model, step, total_steps
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
def training_loss(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    main_loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    aux_logits = getattr(model, "_aux_logits", None)
    if aux_logits is None:
        return main_loss
    mean_logits, terminal_logits = aux_logits
    mean_loss = F.cross_entropy(
        mean_logits, labels, label_smoothing=0.03
    )
    terminal_loss = F.cross_entropy(
        terminal_logits, labels, label_smoothing=0.03
    )
    return (main_loss + 0.1 * mean_loss + 0.1 * terminal_loss) / 1.2
>>>>>>> REPLACE