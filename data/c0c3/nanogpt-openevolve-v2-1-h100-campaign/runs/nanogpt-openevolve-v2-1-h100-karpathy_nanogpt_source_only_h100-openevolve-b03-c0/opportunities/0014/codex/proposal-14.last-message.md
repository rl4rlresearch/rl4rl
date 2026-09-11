MECHANISM: Remove vocabulary-wide logit softcapping

HYPOTHESIS: Eliminating the nearly identity `tanh` softcap will increase throughput beyond 486.2M tokens while preserving stable cross-entropy training, achieving `val_bpb < 0.986636`.

INTENDED_EDIT: Compute fp32 logits directly without scaling, `tanh`, and rescaling over every token-vocabulary element.

EVIDENCE: The best 96-sequence design reached `val_bpb 0.986636` at 486.2M tokens, while capacity-reducing depth and MLP changes regressed; this targets avoidable output-layer computation without reducing model capacity or changing the validated optimizer schedule.

<<<<<<< SEARCH
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x).float()
>>>>>>> REPLACE