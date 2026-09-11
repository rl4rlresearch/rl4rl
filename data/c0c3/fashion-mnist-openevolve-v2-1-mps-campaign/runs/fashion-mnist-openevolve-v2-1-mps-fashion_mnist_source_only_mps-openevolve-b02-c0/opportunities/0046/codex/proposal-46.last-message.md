MECHANISM: Efficient post-ensemble logit sharpening

HYPOTHESIS: Scaling the ensemble log-probabilities by 1.10 will preserve exactly 9,252 correct predictions while lowering validation cross-entropy below 0.2287977, and removing the unused evaluation forward pass will allow the previously timing-out calibration test to complete.

INTENDED_EDIT: Avoid computing discarded unaugmented logits during evaluation and increase the inference-only ensemble scale from 1.05 to 1.10.

EVIDENCE: Scaling from 1.00 to 1.05 preserved 9,252 correct and lowered cross-entropy from 0.2356299 to 0.2287977; subsequent sharpening trials timed out, while the current evaluation path performs one full model pass whose logits are discarded.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.05 * ensemble_log_probabilities
=======
        return 1.10 * ensemble_log_probabilities
>>>>>>> REPLACE