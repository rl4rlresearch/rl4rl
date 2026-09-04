MECHANISM: Annealed auxiliary supervision for the statistics branch

HYPOTHESIS: Independently supervising the statistics head early and annealing that auxiliary loss to zero will exceed 9,345 correct predictions by accelerating robust first-order feature learning while preserving late-stage residual specialization.

INTENDED_EDIT: Expose statistics logits during training and add a computationally negligible auxiliary cross-entropy whose weight decreases linearly from 0.15 to zero; inference, parameters, and example accounting remain unchanged.

EVIDENCE: The first-order statistics bypass improved the classifier, while the attempted higher learning rate for its late-starting residual parameters timed out; direct annealed supervision tests the same underexposure hypothesis without adding parameters, forward passes, or meaningful runtime.

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
=======
    def _forward_once(
        self,
        images: torch.Tensor,
        return_statistics_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(gated_feature_map) + residual_logits
=======
        logits = self.classifier(gated_feature_map) + residual_logits
        if return_statistics_logits:
            return logits, residual_logits
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    logits, statistics_logits = model._forward_once(
        images, return_statistics_logits=True
    )
    progress = step / max(total_steps - 1, 1)
    auxiliary_weight = 0.15 * (1.0 - progress)
    return (
        F.cross_entropy(logits, labels)
        + auxiliary_weight
        * F.cross_entropy(statistics_logits, labels)
    )
>>>>>>> REPLACE