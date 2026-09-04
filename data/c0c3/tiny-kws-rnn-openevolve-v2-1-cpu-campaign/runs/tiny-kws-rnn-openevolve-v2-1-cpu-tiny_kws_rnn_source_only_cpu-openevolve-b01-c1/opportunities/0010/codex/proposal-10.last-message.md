MECHANISM: Auxiliary-trained confidence-gated recurrent checkpoints

HYPOTHESIS: Selective exits at steps 24 and 28 will preserve at least 85% accuracy while reducing mean recurrent steps and total MACs, because only high-confidence examples stop early while ambiguous examples retain all 32 frames.

INTENDED_EDIT: Keep the verified 92-unit GRU, train its shared classifier on 24-, 28-, and 32-step summaries, cache logits only at those checkpoints, and exit at conservative confidence thresholds.

EVIDENCE: Uniform 16- and 24-frame execution failed at 83.19%, while the full 32-step 92-unit GRU reached 85.89%; this challenges the load-bearing assumption that every example needs the same temporal budget without again discarding later evidence for every recording.

<<<<<<< SEARCH
class KeywordGRU(nn.Module):
    """A one-layer causal GRU with an online temporal summary."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
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
    """A causal GRU with trained anytime predictions at sparse checkpoints."""

    decision_steps = (24, 28, 32)

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)
        self._early_logits = None

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        score = torch.zeros(batch_size, 8, device=device, dtype=dtype)
        return hidden, summary, count, score

    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, _score = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        new_summary = summary + output[:, 0, :]
        new_count = count + 1.0
        averaged = new_summary / new_count.clamp_min(1.0)
        completed = int(round(float(new_count[0, 0].detach().item())))

        if completed in self.decision_steps:
            score = self.classifier(averaged)
        else:
            score = 6.0 * averaged[:, :8]

        if self.training:
            if completed == 1:
                self._early_logits = []
            if completed in (24, 28) and isinstance(self._early_logits, list):
                self._early_logits.append(score)

        return hidden.transpose(0, 1), new_summary, new_count, score

    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, count, _score = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        prefix_sums = outputs.cumsum(dim=1)
        new_summary = summary + prefix_sums[:, -1, :]
        new_count = count + frames.shape[1]
        averaged = new_summary / new_count.clamp_min(1.0)

        start = int(round(float(count[0, 0].detach().item())))
        end = start + frames.shape[1]
        checkpoints = [
            checkpoint
            for checkpoint in self.decision_steps
            if start < checkpoint <= end
        ]

        checkpoint_logits = None
        if checkpoints:
            checkpoint_features = []
            for checkpoint in checkpoints:
                offset = checkpoint - start - 1
                checkpoint_summary = summary + prefix_sums[:, offset, :]
                checkpoint_count = count + float(offset + 1)
                checkpoint_features.append(
                    checkpoint_summary / checkpoint_count.clamp_min(1.0)
                )
            checkpoint_logits = self.classifier(
                torch.stack(checkpoint_features, dim=1)
            )

        if end in checkpoints:
            score = checkpoint_logits[:, checkpoints.index(end), :]
        else:
            score = 6.0 * averaged[:, :8]

        if self.training:
            early = []
            if checkpoint_logits is not None:
                early = [
                    checkpoint_logits[:, index, :]
                    for index, checkpoint in enumerate(checkpoints)
                    if checkpoint in (24, 28)
                ]
            self._early_logits = (
                torch.stack(early, dim=1) if early else None
            )

        return hidden.transpose(0, 1), new_summary, new_count, score

    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        return state[3]

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))

    def exit_mask(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step, total_steps
        completed = int(round(float(state[2][0, 0].detach().item())))
        if completed not in (24, 28):
            return torch.zeros(
                logits.shape[0], device=logits.device, dtype=torch.bool
            )
        confidence = logits.softmax(dim=-1).amax(dim=-1)
        threshold = 0.985 if completed == 24 else 0.970
        return confidence >= threshold
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
    loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
    early_logits = getattr(model, "_early_logits", None)
    if isinstance(early_logits, list):
        early_logits = (
            torch.stack(early_logits, dim=1) if early_logits else None
        )
    if early_logits is not None:
        early_labels = labels[:, None].expand(
            -1, early_logits.shape[1]
        ).reshape(-1)
        early_loss = F.cross_entropy(
            early_logits.reshape(-1, 8),
            early_labels,
            label_smoothing=0.03,
        )
        loss = loss + 0.20 * early_loss
    return loss
>>>>>>> REPLACE