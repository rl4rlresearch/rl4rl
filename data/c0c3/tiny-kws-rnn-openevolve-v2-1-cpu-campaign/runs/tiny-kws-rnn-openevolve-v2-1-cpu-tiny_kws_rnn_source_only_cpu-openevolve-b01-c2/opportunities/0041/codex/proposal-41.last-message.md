MECHANISM: Confidence-gated penultimate early exit

HYPOTHESIS: Exiting examples whose penultimate-step posterior is at least 0.95 will retain at least 85% validation accuracy and reduce total inference MACs whenever more than 8.75% of examples exit before the final GRU step.

INTENDED_EDIT: Evaluate the learned classifier only at steps 21 and 22, and add a high-confidence exit after step 21 while preserving the qualified 22-frame schedule and final classifier behavior.

EVIDENCE: The qualified 22-step design reached 85.40%, while unconditional removal of one step fell to 84.29%; selectively skipping the final step only for already-confident examples is a narrower temporal reduction that preserves all 22 steps for uncertain examples.

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
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
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        if bool((count < 21.0).all()):
            return hidden.new_zeros(hidden.shape[0], 8)
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

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
        count = state[3][:, 0]
        confidence = logits.softmax(dim=1).amax(dim=1)
        return (count == 21.0) & (confidence >= 0.95)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]
>>>>>>> REPLACE