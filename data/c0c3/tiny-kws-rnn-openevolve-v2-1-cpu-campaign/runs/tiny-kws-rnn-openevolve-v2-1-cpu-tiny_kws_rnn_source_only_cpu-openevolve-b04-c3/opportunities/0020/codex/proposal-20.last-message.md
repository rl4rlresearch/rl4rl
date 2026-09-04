MECHANISM: Dual-view consensus-gated anytime inference

HYPOTHESIS: Independently supervised mean-state and terminal-state predictors can safely exit highly confident, agreeing examples after 23 steps, retaining at least 85% accuracy while reducing exact inference MACs below 1,198,151,875.

INTENDED_EDIT: Replace the inseparable concatenated readout with independently supervised temporal-mean and terminal heads, add a cheap early probe, and stop examples up to four frames early only when both full heads confidently agree.

EVIDENCE: The 125-unit dual-readout model qualified at 85.52%, and adding terminal information previously lifted the 127-unit/27-step design from 84.91% to 85.40%. This challenges the shared fixed-compute assumption while preserving the qualified recurrent path and making its two predictive views usable as a conservative exit consensus.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(250, 8)
=======
        self.gru = nn.GRU(20, 125, num_layers=1, batch_first=True)
        self.mean_classifier = nn.Linear(125, 8)
        self.terminal_classifier = nn.Linear(125, 8)
        self.early_probe = nn.Linear(16, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        features = torch.cat(
            (summary / count.clamp_min(1.0), hidden[:, 0, :]), dim=-1
        )
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        terminal = hidden[:, 0, :]
        probe_logits = self.early_probe(terminal[:, :16])
        self._probe_logits = probe_logits

        if not self.training and count.max().item() < 23.0:
            return probe_logits

        mean_logits = self.mean_classifier(
            summary / count.clamp_min(1.0)
        )
        terminal_logits = self.terminal_classifier(terminal)
        self._mean_logits = mean_logits
        self._terminal_logits = terminal_logits
        return mean_logits + terminal_logits

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        if total_steps <= 23:
            return torch.zeros(
                logits.shape[0], device=logits.device, dtype=torch.bool
            )

        prediction = logits.argmax(dim=-1)
        mean_probabilities = torch.softmax(self._mean_logits, dim=-1)
        terminal_probabilities = torch.softmax(
            self._terminal_logits, dim=-1
        )
        combined_confidence = torch.softmax(logits, dim=-1).amax(dim=-1)
        agreement = (
            (self._mean_logits.argmax(dim=-1) == prediction)
            & (self._terminal_logits.argmax(dim=-1) == prediction)
        )
        return (
            (count[:, 0] >= 23.0)
            & (count[:, 0] < float(total_steps))
            & agreement
            & (combined_confidence >= 0.995)
            & (mean_probabilities.amax(dim=-1) >= 0.90)
            & (terminal_probabilities.amax(dim=-1) >= 0.90)
        )

    def frame_schedule(self, available_frames: int) -> list[int]:
>>>>>>> REPLACE

<<<<<<< SEARCH
    del model, step, total_steps
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    del step, total_steps
    main_loss = F.cross_entropy(
        logits, labels, label_smoothing=0.03
    )
    view_loss = 0.5 * (
        F.cross_entropy(
            model._mean_logits, labels, label_smoothing=0.03
        )
        + F.cross_entropy(
            model._terminal_logits, labels, label_smoothing=0.03
        )
    )
    probe_loss = F.cross_entropy(
        model._probe_logits, labels, label_smoothing=0.03
    )
    return main_loss + 0.10 * view_loss + 0.02 * probe_loss
>>>>>>> REPLACE