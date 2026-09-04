MECHANISM: Moderately stronger finite-logit confidence regularization

HYPOTHESIS: Restoring the proven 1024-token short window and using a softcap of 9 will lower val_bpb below 0.995334 by strengthening the beneficial cap-10 regularization without the excessive restriction observed at cap 7.5.

INTENDED_EDIT: Restore half-context sliding attention and change the FP32 training-and-validation logit softcap from 15 to 9.

EVIDENCE: Half-context cap 10 achieved the best observed val_bpb of 0.995334 on 493.4M tokens; quarter-context attention regressed to 1.009577, while cap 7.5 regressed to 1.013318, motivating a narrowly intermediate softcap on the proven architecture.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 2
>>>>>>> REPLACE

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x).float()
        softcap = 9.0
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE