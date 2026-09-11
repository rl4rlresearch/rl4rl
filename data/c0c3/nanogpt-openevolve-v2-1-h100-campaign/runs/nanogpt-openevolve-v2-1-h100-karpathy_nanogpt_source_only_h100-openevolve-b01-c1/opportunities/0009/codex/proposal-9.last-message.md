MECHANISM: Raw-logit cross-entropy

HYPOTHESIS: Removing the vocabulary-wide tanh softcap will increase throughput while preserving the 50.3M-parameter model and essential final global-attention layer, lowering val_bpb below 0.992286.

INTENDED_EDIT: Feed FP32 logits directly to cross-entropy, eliminating the softcap division, tanh, and multiplication.

EVIDENCE: Grouped-query attention processed 16.8M more tokens but regressed to 0.996550 after reducing capacity; this motivates a throughput optimization that leaves the successful attention architecture and parameter count intact.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x)
        logits = logits.float()
>>>>>>> REPLACE