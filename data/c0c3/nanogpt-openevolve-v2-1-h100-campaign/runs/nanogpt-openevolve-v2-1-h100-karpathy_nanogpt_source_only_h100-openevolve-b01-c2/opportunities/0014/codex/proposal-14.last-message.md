MECHANISM: Uncapped vocabulary loss on the proven high-throughput batch

HYPOTHESIS: Removing the vocabulary-wide tanh softcap while restoring the verified 524K-token batch will process more than 497M tokens and lower val_bpb below 0.995558, because logits below the cap retain nearly identical behavior while avoiding a large elementwise operation.

INTENDED_EDIT: Restore two-microbatch gradient accumulation and train/evaluate directly on float logits without the softcap transform.

EVIDENCE: The 524K-token SSSL design achieved the best val_bpb, 0.995558, on 497.0M tokens, while reducing Muon polar iterations did not materially increase throughput; this motivates optimizing the per-token vocabulary path instead of further reducing optimizer work.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
=======
        logits = self.lm_head(x).float()

        if targets is not None:
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # 262K tokens; one 128-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 2**19 # 524K tokens; two 128-sequence microbatches per step
>>>>>>> REPLACE