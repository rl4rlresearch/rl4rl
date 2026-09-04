MECHANISM: Class-specific smooth maxout decision head

HYPOTHESIS: Replacing each class’s single affine logit with a smooth mixture of three class-specific affine modes will exceed 9,286 correct predictions by learning multiple decision regions per class without the runtime and optimization costs of widening the shared representation.

INTENDED_EDIT: Replace the single 48-to-10 output layer with a three-mode-per-class log-mean-exp head. This challenges the existing assumption that every class is adequately represented by one linear decision template after the shared feature bottleneck.

EVIDENCE: Widening the shared hidden layer to 54 units regressed to 9,254 correct, while global-attention prediction timed out; meanwhile, mirrored-view ensembling improved accuracy, suggesting value in alternative class evidence. The proposed head adds only 980 parameters and negligible computation while testing class-specific multimodal prediction directly.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0


class MixtureLogitHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        modes_per_class: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.modes_per_class = modes_per_class
        self.projections = nn.Linear(
            in_features,
            num_classes * modes_per_class,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mode_logits = self.projections(features).view(
            features.shape[0],
            self.num_classes,
            self.modes_per_class,
        )
        return torch.logsumexp(mode_logits, dim=-1) - math.log(
            self.modes_per_class
        )


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Dropout(0.15),
            nn.Linear(48, 10),
=======
            nn.Dropout(0.15),
            MixtureLogitHead(48, 10, modes_per_class=3),
>>>>>>> REPLACE