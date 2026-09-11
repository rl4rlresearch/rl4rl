MECHANISM: Intermediate finite-logit confidence regularization

HYPOTHESIS: Raising the softcap from 10 to 11 will preserve beneficial confidence control while slightly reducing saturation, lowering val_bpb below 0.995334.

INTENDED_EDIT: Change the FP32 training-and-validation logit softcap from 10 to 11, leaving architecture, batching, and optimization unchanged.

EVIDENCE: Cap 10 achieved the best observed val_bpb of 0.995334, outperforming cap 9 at 0.995704 and cap 15 at 0.995558; cap 11 probes the narrower, more promising side of the apparent optimum.

<<<<<<< SEARCH
        softcap = 10.0
        logits = softcap * torch.tanh(logits / softcap)
=======
        softcap = 11.0
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE