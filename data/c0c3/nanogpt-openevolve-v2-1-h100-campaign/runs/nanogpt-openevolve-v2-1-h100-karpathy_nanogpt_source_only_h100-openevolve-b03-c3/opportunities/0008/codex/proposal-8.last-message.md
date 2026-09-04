MECHANISM: Intermediate tanh confidence regularization

HYPOTHESIS: A softcap of 12 will lower val_bpb below 0.995334 at comparable throughput by relaxing cap 10 slightly without approaching the weaker regularization of cap 15.

INTENDED_EDIT: Change the FP32 training-and-validation logit softcap from 9 to 12, leaving architecture, batching, and optimization unchanged.

EVIDENCE: Cap 10 achieved 0.995334, while full-throughput cap 15 reached 0.995558 versus 0.995704 for cap 9, suggesting the more promising side of cap 10 is upward; the cap-11 run trained only 394.8M tokens versus 493.4M and therefore did not isolate that region.

<<<<<<< SEARCH
        logits = self.lm_head(x).float()
        softcap = 9.0
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x).float()
        softcap = 12.0
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE