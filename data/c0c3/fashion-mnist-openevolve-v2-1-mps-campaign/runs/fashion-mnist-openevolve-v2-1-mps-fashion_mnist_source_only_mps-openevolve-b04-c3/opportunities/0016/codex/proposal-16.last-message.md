MECHANISM: Higher-step-equivalent cosine optimization

HYPOTHESIS: Restoring logit-space flip averaging and increasing the peak learning rate by 20% will exceed 9,229 correct predictions by translating the proven benefit of more optimizer updates into greater optimization distance without the timeout risk of smaller batches.

INTENDED_EDIT: Restore the qualified logit-space flip ensemble and raise the AdamW cosine schedule’s peak learning rate from 2.5e-3 to 3.0e-3 while retaining batch size 64 and all other training behavior.

EVIDENCE: Batch size 64 improved correctness from 9,210 to 9,229 by doubling optimizer steps, but smaller batches timed out; probability-space fusion then reduced correctness to 9,227. This motivates returning to qualified inference and testing a compute-neutral increase in optimization distance.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=-1),
                F.log_softmax(flipped_logits, dim=-1),
            ) - math.log(2.0)
        return logits
=======
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1e-4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 3.0e-3 * multiplier
>>>>>>> REPLACE