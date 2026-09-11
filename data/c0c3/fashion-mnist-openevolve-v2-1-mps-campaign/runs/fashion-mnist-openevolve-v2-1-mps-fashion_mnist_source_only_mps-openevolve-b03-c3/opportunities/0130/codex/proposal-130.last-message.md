MECHANISM: Residual multi-prototype maxout class scoring

HYPOTHESIS: Adding three learned residual prototypes per class to the 9,284-correct mixed-pooling design will exceed 9,284 correct predictions by modeling class-specific appearance modes without disturbing the proven shared representation or base linear classifier.

INTENDED_EDIT: Restore the best verified mixed pooling and beta2=0.96, then replace each class’s single affine score with a preserved base score plus the strongest of three learned residual evidence templates.

EVIDENCE: Learnable mixed pooling achieved the best result at 9,284 correct, while widening the shared classifier fell to 9,239; this challenges the load-bearing assumption that added capacity should be shared and instead allocates the remaining parameter budget to conditional, class-specific prediction paths.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
GRAD_CLIP_NORM = 5.0


class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average


class MultiPrototypeClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        classes: int,
        residual_prototypes: int = 3,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.residual_prototypes = residual_prototypes
        self.base = nn.Linear(in_features, classes)
        self.residuals = nn.Linear(
            in_features, classes * residual_prototypes
        )
        nn.init.normal_(self.residuals.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residuals.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        base_logits = self.base(features)
        residual_logits = self.residuals(features).view(
            features.size(0), self.classes, self.residual_prototypes
        )
        zero_prototype = residual_logits.new_zeros(
            residual_logits.size(0), self.classes, 1
        )
        correction = torch.cat(
            (zero_prototype, residual_logits), dim=-1
        ).amax(dim=-1)
        return base_logits + correction


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Dropout(0.10),
            nn.Linear(140, 10),
=======
            nn.Dropout(0.10),
            MultiPrototypeClassifier(140, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
        betas=(0.9, 0.95),
=======
        betas=(0.9, 0.96),
>>>>>>> REPLACE