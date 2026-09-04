MECHANISM: Full-batch near-optimal logit confidence regularization

HYPOTHESIS: A softcap of 11.75 on the proven 524,288-token batch will lower val_bpb below 0.995230 by refining the apparent optimum just below cap 12.

INTENDED_EDIT: Restore two-microbatch gradient accumulation and change the training-and-validation FP32 logit softcap from 15 to 11.75.

EVIDENCE: Cap 12 achieved the best val_bpb of 0.995230, outperforming cap 10 at 0.995334 and cap 15 at 0.995558; meanwhile, halving the batch reduced trained tokens to 376.2M and regressed val_bpb to 1.001276.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x).float()
        softcap = 11.75
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
>>>>>>> REPLACE