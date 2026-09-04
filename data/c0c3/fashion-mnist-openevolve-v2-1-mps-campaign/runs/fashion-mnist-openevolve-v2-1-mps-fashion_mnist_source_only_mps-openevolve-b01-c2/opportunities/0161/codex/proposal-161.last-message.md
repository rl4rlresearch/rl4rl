MECHANISM: Confidence-adaptive positive logit calibration

HYPOTHESIS: A positive scale that increases slightly for above-average-confidence images and decreases for below-average-confidence images will preserve exactly 9,331 argmax predictions while reducing validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Retain the best verified flip fusion and global scale, then add a small batch-centered, per-image confidence adjustment that cannot change predicted classes.

EVIDENCE: Both adjacent float32 global scales worsened cross-entropy at the same 9,331 correct, indicating global calibration is locally optimized; confidence-conditional scaling tests an orthogonal calibration dimension while mathematically preserving every argmax.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
=======
            logits = (
                0.4914990234375 * logits
                + 0.5085009765625 * flipped_logits
            )
            confidence = F.softmax(logits, dim=1).amax(dim=1, keepdim=True)
            adaptive_scale = 1.184 * (
                1.0 + 0.05 * (confidence - confidence.mean())
            )
            logits = adaptive_scale * logits
>>>>>>> REPLACE