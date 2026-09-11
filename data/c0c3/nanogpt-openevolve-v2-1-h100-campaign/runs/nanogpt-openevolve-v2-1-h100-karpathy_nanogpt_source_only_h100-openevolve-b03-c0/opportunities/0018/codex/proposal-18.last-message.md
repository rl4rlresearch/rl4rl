MECHANISM: Stronger logit softcapping

HYPOTHESIS: Lowering the logit softcap from 15 to 10 will improve regularization and achieve `val_bpb < 0.986491` without changing model capacity, batching, or training throughput.

INTENDED_EDIT: Apply a stronger tanh softcap to the output logits while preserving the best max-autotuned implementation.

EVIDENCE: Removing the softcap regressed `val_bpb` sharply from `0.986491` to `0.993511` at nearly identical throughput, demonstrating that logit compression materially improves generalization and motivating a directional test of stronger compression.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        softcap = 10
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE