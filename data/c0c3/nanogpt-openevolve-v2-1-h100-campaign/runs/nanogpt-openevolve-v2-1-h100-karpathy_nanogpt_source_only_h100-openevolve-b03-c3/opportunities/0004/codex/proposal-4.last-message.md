MECHANISM: Stronger finite-logit confidence regularization

HYPOTHESIS: Applying a tanh softcap of 10 will reduce val_bpb below 0.995558 by more strongly suppressing overconfident logits than the successful cap of 15.

INTENDED_EDIT: Replace uncapped FP32 logits with softcapped FP32 logits during both training and validation.

EVIDENCE: Removing the cap increased tokens from 497.0M to 500.7M but worsened val_bpb from 0.995558 to 1.000923, showing that finite-logit regularization matters more than its small throughput cost and motivating a stronger cap.

<<<<<<< SEARCH
        logits = self.lm_head(x).float()

        if targets is not None:
=======
        logits = self.lm_head(x).float()
        softcap = 10.0
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
>>>>>>> REPLACE