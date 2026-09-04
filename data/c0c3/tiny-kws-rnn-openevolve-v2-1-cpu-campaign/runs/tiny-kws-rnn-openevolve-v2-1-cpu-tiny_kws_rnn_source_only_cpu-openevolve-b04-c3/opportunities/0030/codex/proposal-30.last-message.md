MECHANISM: Hierarchical recurrent factorization with multilevel temporal readout

HYPOTHESIS: A stacked 48-unit acoustic GRU and 90-unit contextual GRU, using all 27 qualified frames and both levels’ mean and terminal states, will retain at least 85% accuracy while reducing total inference MACs below the qualified 120-unit/26-step model’s 1.070B.

INTENDED_EDIT: Replace the monolithic 120-unit recurrence with two narrower recurrent stages, normalize their interface and readouts, and classify from fine- and context-scale temporal summaries while retaining training-only dual-view supervision.

EVIDENCE: The 120-unit model qualified at 86.13% with 27 steps, but reducing either temporal coverage or width approached the accuracy boundary. This suggests monolithic width and frame deletion are load-bearing assumptions. The factorized recurrence preserves 27-frame coverage and increases total recurrent state from 120 to 138 units while reducing structural recurrent cost from 50,400 to 47,052 MACs per frame.

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
>>>>>>> REPLACE