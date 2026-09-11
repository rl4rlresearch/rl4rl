MECHANISM: Bounded-state certified early exit

HYPOTHESIS: Starting exact classifier checks at recurrent step 20 and exiting only when the current class is mathematically invariant to every possible remaining bounded GRU output will preserve validation accuracy at or above 85% while reducing dense MACs and mean recurrent steps below the verified 30-step model.

INTENDED_EDIT: Keep the verified model and training unchanged, defer learned readout until step 20, then use classifier-weight bounds to skip remaining frames only when they cannot alter the predicted class.

EVIDENCE: The 30-step model achieved 85.03%, while both fixed 29-step boundary schedules achieved 84.66%; this motivates retaining the final step for ambiguous recordings while safely omitting it—and potentially additional late steps—for recordings whose decision is already provably fixed.

<<<<<<< SEARCH
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.gru = nn.GRU(18, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
        self.exit_readout_start = 20
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        mean_summary = summary / count.clamp_min(1.0)
        if bool((count[:, 0] < float(self.exit_readout_start)).all()):
            return mean_summary[:, :8]
        return self.classifier(mean_summary)

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        executed = count[:, 0]
        eligible = (
            (executed >= float(self.exit_readout_start))
            & (executed >= 2.0)
            & (executed < float(total_steps))
        )
        if not bool(eligible.any()):
            return torch.zeros_like(eligible)

        winner = logits.argmax(dim=-1)
        margins = logits.gather(1, winner.unsqueeze(1)) - logits

        weight = self.classifier.weight
        winner_weight = weight.index_select(0, winner)
        weight_delta = winner_weight.unsqueeze(1) - weight.unsqueeze(0)
        future_sensitivity = 1.0001 * weight_delta.abs().sum(dim=-1)

        bias = self.classifier.bias
        winner_bias = bias.index_select(0, winner)
        bias_delta = winner_bias.unsqueeze(1) - bias.unsqueeze(0)

        remaining = float(total_steps) - executed
        lower_final_margin = (
            executed.unsqueeze(1) * margins
            + remaining.unsqueeze(1) * (bias_delta - future_sensitivity)
        )
        class_ids = torch.arange(
            logits.shape[1], device=logits.device
        ).unsqueeze(0)
        competitors_safe = (
            (lower_final_margin > 1.0e-5)
            | (class_ids == winner.unsqueeze(1))
        )
        return eligible & competitors_safe.all(dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))
>>>>>>> REPLACE