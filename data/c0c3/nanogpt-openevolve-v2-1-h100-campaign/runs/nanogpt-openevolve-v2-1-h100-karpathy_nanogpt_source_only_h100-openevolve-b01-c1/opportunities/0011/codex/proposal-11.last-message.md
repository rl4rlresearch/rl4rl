MECHANISM: BF16 logit softcapping before FP32 cross-entropy

HYPOTHESIS: Computing the retained tanh softcap in BF16 will reduce vocabulary-wide elementwise cost without materially changing its regularizing effect, lowering val_bpb below 0.992286.

INTENDED_EDIT: Apply the logit softcap before converting logits to FP32 for cross-entropy.

EVIDENCE: Removing the softcap regressed val_bpb from 0.992286 to 1.012563, so this preserves the beneficial operation while targeting its FP32 computational cost.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        logits = logits.float()
>>>>>>> REPLACE