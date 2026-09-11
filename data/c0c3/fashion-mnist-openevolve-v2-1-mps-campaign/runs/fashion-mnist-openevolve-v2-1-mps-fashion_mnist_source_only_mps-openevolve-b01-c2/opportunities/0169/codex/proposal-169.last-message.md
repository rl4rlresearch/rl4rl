MECHANISM: Rank-preserving inverse-margin calibration

HYPOTHESIS: Slightly increasing temperature for high-margin predictions and decreasing it for low-margin predictions will preserve all 9,331 argmax predictions while lowering validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Replace the fixed evaluation scale with a batch-centered, margin-conditioned positive scale while retaining the best verified flip-fusion weights and unchanged training.

EVIDENCE: Both adjacent global scales worsened cross-entropy without changing correctness, indicating that 1.184 is locally optimal globally; the prior confidence-adaptive experiment timed out, leaving conditional calibration untested, while inverse-margin scaling specifically softens costly confident errors and strengthens uncertain correct predictions.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            mixed_logits = (
                0.4914990234375 * logits
                + 0.5085009765625 * flipped_logits
            )
            top_two = mixed_logits.topk(k=2, dim=1).values
            margin = top_two[:, 0] - top_two[:, 1]
            adaptive_scale = (
                1.0 - 0.005 * (margin - margin.mean())
            ).clamp(min=0.9, max=1.1)
            logits = 1.184 * adaptive_scale.unsqueeze(1) * mixed_logits
        return logits
>>>>>>> REPLACE