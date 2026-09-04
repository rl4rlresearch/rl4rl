MECHANISM: Uncapped fused vocabulary loss

HYPOTHESIS: Removing the vocabulary-wide tanh softcap will increase trained tokens without destabilizing the five-minute run and reduce val_bpb below 0.995558.

INTENDED_EDIT: Replace softcapped logits with direct FP32 logits for both training and validation.

EVIDENCE: Quarter-context attention regressed from 0.995558 on 497.0M tokens to 1.009577 on 401.1M tokens, indicating that reducing attention work did not improve throughput; eliminating the dense tanh forward/backward computation targets a different per-token cost while leaving model capacity and context unchanged.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x).float()
>>>>>>> REPLACE