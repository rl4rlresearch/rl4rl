MECHANISM: Dual-statistic coarse-grid representation

HYPOTHESIS: Replacing the coordinate-sensitive 7×7 dense bottleneck with 4×4 average/max feature grids and widening the convolutional channels will exceed 9,349 correct predictions by combining preserved spatial layout with greater translation stability and richer learned features.

INTENDED_EDIT: Replace the 32/64-channel backbone and flattened 7×7 classifier with a 40/80-channel backbone and a learned head over concatenated 4×4 average- and max-pooled feature grids, remaining below the parameter ceiling at 247,350 parameters.

EVIDENCE: Global pooling reduced correctness to 9,085, proving that layout must remain available, while translation fusion raised the current classifier to 9,349, indicating harmful sensitivity to exact coordinates. A coarse spatial grid retains layout without the unrestricted fine-coordinate dependence of the current dense head and frees parameters to widen the feature extractor.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(56, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
class DualStatisticGridHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * channels * 4 * 4, 52),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(52, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_grid = F.adaptive_avg_pool2d(features, output_size=4)
        maximum_grid = F.adaptive_max_pool2d(features, output_size=4)
        descriptor = torch.cat((average_grid, maximum_grid), dim=1)
        return self.classifier(descriptor.flatten(1))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.Conv2d(40, 40, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.Conv2d(80, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.MaxPool2d(2),
            SpatialRefinement(80),
        )
        self.classifier = DualStatisticGridHead(80)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
>>>>>>> REPLACE