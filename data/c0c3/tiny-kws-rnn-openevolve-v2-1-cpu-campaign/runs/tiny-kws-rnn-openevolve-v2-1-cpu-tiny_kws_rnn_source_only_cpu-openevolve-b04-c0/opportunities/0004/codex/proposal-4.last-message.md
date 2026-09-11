MECHANISM: Certified adaptive early exit

HYPOTHESIS: Worst-case logit-margin certification will preserve the 104-unit GRU’s full-sequence class decisions while reducing mean recurrent steps below 32 and therefore lowering exact inference MACs.

INTENDED_EDIT: Add an `exit_mask` that exits only when the current winning class cannot be overturned by any possible remaining GRU outputs, using the GRU’s bounded hidden range and classifier weights.

EVIDENCE: The 104-unit full-resolution model barely met the target at 85.03%, while fixed 16-frame decimation fell to 81.72%; this motivates preserving all steps for uncertain examples and skipping computation only when the final class is certified unchanged.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        count = count[:, 0]
        remaining = (float(total_steps) - count).clamp_min(0.0)

        winners = logits.argmax(dim=1)
        winner_weight = self.classifier.weight[winners]
        winner_bias = self.classifier.bias[winners]

        weight_delta = (
            winner_weight.unsqueeze(1)
            - self.classifier.weight.unsqueeze(0)
        )
        bias_delta = (
            winner_bias.unsqueeze(1)
            - self.classifier.bias.unsqueeze(0)
        )
        current_margin = (
            logits.gather(1, winners.unsqueeze(1)) - logits
        )

        # Every GRU output coordinate lies in [-1, 1]. This is therefore a
        # lower bound on each final winner-versus-competitor logit margin
        # after all unexecuted frames, regardless of their values.
        final_margin_lower_bound = (
            count.unsqueeze(1) * current_margin
            + remaining.unsqueeze(1)
            * (bias_delta - weight_delta.abs().sum(dim=2))
        )

        classes = torch.arange(logits.shape[1], device=logits.device)
        competitors = classes.unsqueeze(0) != winners.unsqueeze(1)
        certified = (
            (final_margin_lower_bound > 0.0) | ~competitors
        ).all(dim=1)
        return (count >= 2.0) & (remaining > 0.0) & certified
>>>>>>> REPLACE