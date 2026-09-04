MECHANISM: Probability-space test-time view ensemble

HYPOTHESIS: Arithmetic probability fusion of the original and flipped views will exceed 9,331 correct predictions, or retain 9,331 while reducing cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Restore the best verified view weights and combine normalized view probabilities instead of raw logits, preserving training, parameters, and inference cost.

EVIDENCE: Neighboring logit-fusion weights and calibration scales have saturated at 9,331 correct, while evaluation-aligned training and EMA reduced correctness; probability fusion is an orthogonal inference-only ensemble rule that limits domination by an overconfident disagreeing view.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (
                0.4914921875 * logits + 0.5085078125 * flipped_logits
            )
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            original_log_probs = F.log_softmax(logits, dim=1)
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
            logits = 1.184 * torch.logaddexp(
                original_log_probs + math.log(0.4914990234375),
                flipped_log_probs + math.log(0.5085009765625),
            )
        return logits
>>>>>>> REPLACE