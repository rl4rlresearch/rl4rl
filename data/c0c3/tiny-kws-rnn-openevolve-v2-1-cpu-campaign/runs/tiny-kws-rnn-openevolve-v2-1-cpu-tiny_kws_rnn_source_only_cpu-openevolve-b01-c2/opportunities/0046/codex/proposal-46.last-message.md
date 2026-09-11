MECHANISM: Confidence-gated penultimate early exit

HYPOTHESIS: On the qualified 22-frame schedule, exiting examples with at least 0.95 penultimate-step posterior confidence will preserve validation accuracy at or above 85% and reduce inference MACs if more than approximately 8.75% of examples skip the final GRU step.

INTENDED_EDIT: Adopt the qualified frames 3–21, 24, 26, and 28 schedule; execute the learned classifier only for the last two steps; and exit high-confidence examples after step 21.

EVIDENCE: The 22-step schedule achieved 85.40% accuracy, while every tested unconditional 21-step schedule failed. Selective final-step omission therefore targets temporal savings without withholding the final observation from uncertain examples.

<<<<<<< SEARCH
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 9)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
=======
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]

    def exit_mask(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        count = state[3][:, 0]
        confidence = logits.softmax(dim=1).amax(dim=1)
        return (count == float(total_steps - 1)) & (confidence >= 0.95)
>>>>>>> REPLACE