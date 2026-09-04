MECHANISM: Verified batch-64 width-and-update allocation

HYPOTHESIS: Restoring the proven 247,538-parameter batch-64 design will exceed the current design’s 9,214 correct predictions while completing within the verification limit.

INTENDED_EDIT: Replace the wider residual network with the verified 24/48/96-channel model, use batch 64 and its 2.5e-3 schedule, and batch each original/mirrored inference pair into one forward pass.

EVIDENCE: Reference Design 2 achieved the best available result—9,232 correct with 0.2405 cross-entropy—in 58.8 seconds; the current batch-32 residual design achieved only 9,214 correct and took 81.3 seconds.

<<<<<<< SEARCH
BATCH_SIZE = 32
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(28, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(56, 112, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(112),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(112, 112, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(112),
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112 * 3 * 3, 64),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stage1(images)
        features = self.stage2(features)
        features = self.stage3(features)
        features = F.relu(features + self.residual(features), inplace=True)
        return self.classifier(self.pool(features))
=======
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(128, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
>>>>>>> REPLACE

<<<<<<< SEARCH
        log_prob_views = []
        for view in views:
            log_prob_views.append(
                F.log_softmax(self._forward_once(view), dim=1)
            )
            log_prob_views.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
            )
        stacked = torch.stack(log_prob_views, dim=0)
=======
        log_prob_views = []
        for view in views:
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_prob_views.extend((original, mirrored))

        stacked = torch.stack(log_prob_views, dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.0e-3,
        betas=(0.9, 0.99),
=======
        lr=2.5e-3,
        betas=(0.9, 0.99),
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.0e-3 * multiplier
=======
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE