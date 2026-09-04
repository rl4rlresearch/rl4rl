MECHANISM: Training-only dual-view deep supervision

HYPOTHESIS: Independently supervising the mean and terminal recurrent views of a 124-unit, 27-step GRU will recover the two validation examples needed for at least 85% accuracy while retaining the approximately 1.180B inference-MAC cost of the prior 124-unit dual-readout design.

INTENDED_EDIT: Use the lower-cost 124-unit GRU and qualified early-drop schedule, classify from concatenated mean and terminal states, and add two auxiliary classifiers that execute only during training.

EVIDENCE: The prior 124-unit dual-readout model missed qualification by only two examples with 0.3884 cross-entropy, while adding terminal information previously raised the 127-unit design from 84.91% to 85.40%; direct training supervision for both predictive views targets that narrow optimization gap without adding validation MACs.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 127, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(127, 8)
=======
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(248, 8)
        self.mean_aux = nn.Linear(124, 8)
        self.terminal_aux = nn.Linear(124, 8)
        self._aux_logits = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 127, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 127, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
=======
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

    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        schedule = [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]
        if len(schedule) == 28:
            schedule.pop(0)
        return schedule
>>>>>>> REPLACE

<<<<<<< SEARCH
    del model, step, total_steps
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
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