MECHANISM: Penultimate-step certified early exit

HYPOTHESIS: A classifier-weight bound at step 29 will preserve every exited example’s final predicted class, maintaining at least 85% accuracy while reducing mean recurrent steps and exact MACs.

INTENDED_EDIT: Defer the learned readout until the penultimate step, then exit examples whose class cannot change under any possible final bounded GRU output.

EVIDENCE: The 30-step model meets the target at 85.03%, while fixed 29-step schedules narrowly fail; the prior certified-exit attempt timed out without accuracy evidence, motivating this cheaper single-check variant that retains the final step only for uncertified examples.

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
        averaged = summary / count.clamp_min(1.0)
        if bool(torch.all(count < 29.0)):
            return averaged[:, :8]
        return self.classifier(averaged)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(2, available_frames))

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        completed = count[:, 0]
        eligible = completed == float(total_steps - 1)
        if not bool(eligible.any()):
            return eligible

        winner = logits.argmax(dim=1)
        weights = self.classifier.weight.detach()
        biases = self.classifier.bias.detach()
        pairwise_l1 = (
            weights[:, None, :] - weights[None, :, :]
        ).abs().sum(dim=-1)
        uncertainty = pairwise_l1[winner]
        bias_delta = biases[winner, None] - biases[None, :]
        current_margin = logits.gather(1, winner[:, None]) - logits
        remaining = float(total_steps) - completed
        lower_bound = (
            completed[:, None] * current_margin
            + remaining[:, None] * (bias_delta - uncertainty)
        ) / float(total_steps)
        lower_bound = lower_bound.scatter(
            1, winner[:, None], float("inf")
        )
        return eligible & (lower_bound.min(dim=1).values > 0.0)
>>>>>>> REPLACE