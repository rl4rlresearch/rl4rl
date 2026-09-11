MECHANISM: Stronger finite-logit confidence regularization

HYPOTHESIS: On the proven 524,288-token batch, reducing the tanh softcap from 10 to 7.5 will extend the improvement from cap 15 to cap 10 and lower val_bpb below 0.995334.

INTENDED_EDIT: Restore the higher-throughput 524,288-token optimizer batch and apply a 7.5 FP32 logit softcap during training and validation.

EVIDENCE: Cap 10 achieved the best observed val_bpb of 0.995334 versus 0.995558 for cap 15, while the 262,144-token batch regressed to 1.001276 and reduced training volume from roughly 493–497M tokens to 376.2M.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x).float()
        softcap = 7.5
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
>>>>>>> REPLACE