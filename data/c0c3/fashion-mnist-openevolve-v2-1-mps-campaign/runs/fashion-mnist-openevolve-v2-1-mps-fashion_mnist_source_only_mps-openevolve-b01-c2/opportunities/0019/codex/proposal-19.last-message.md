MECHANISM: Cosine flip-loss curriculum

HYPOTHESIS: A cosine shift from equal view/ensemble supervision to pure ensemble supervision will exceed 9,330 correct predictions by preserving view guidance longer early and emphasizing validation-matched ensembling more strongly late.

INTENDED_EDIT: Restore the best batch-64 ungated residual architecture and static 0.02 label smoothing, while replacing linear ensemble-weight annealing with a same-endpoint, same-average cosine curriculum.

EVIDENCE: Linear ensemble weighting achieved the best 9,330 correct versus 9,325 for static 75% weighting, while terminal label-smoothing decay reduced accuracy to 9,322; this isolates the curriculum shape while retaining the proven regularization.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.GELU(),
            nn.Linear(4, 64),
            nn.Sigmoid(),
        )
=======
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = F.gelu(features + self.refine(features))
        channel_scale = 2.0 * self.channel_gate(features)
        features = features * channel_scale.view(-1, 64, 1, 1)
        return self.classifier(features)
=======
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
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
    paired_images = torch.cat(
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.75 - 0.25 * math.cos(math.pi * progress)

    paired_images = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    return 0.5 * (ensemble_loss + view_loss)
=======
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
>>>>>>> REPLACE