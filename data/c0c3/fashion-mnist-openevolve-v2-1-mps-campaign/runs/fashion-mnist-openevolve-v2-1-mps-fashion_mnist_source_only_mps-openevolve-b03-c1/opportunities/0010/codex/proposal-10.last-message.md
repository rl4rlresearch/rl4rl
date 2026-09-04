MECHANISM: Fixed differential-basis image representation

HYPOTHESIS: Exposing raw intensity, horizontal and vertical Sobel responses, and Laplacian detail will exceed 9,091 correct predictions because the fixed exposure budget no longer requires the shallow stem to learn these shape-sensitive primitives from pixels alone.

INTENDED_EDIT: Challenge the raw-pixel-only representation assumption by adding three parameter-free differential channels before the existing CNN and expanding its first convolution from one to four inputs, for approximately 245,034 learned parameters.

EVIDENCE: Doubling optimizer updates left accuracy tied at 9,050 while increasingly refined spatial ensembling produced only incremental gains to 9,091, suggesting optimization and translation variance are no longer the main bottlenecks; a fixed edge/detail basis cleanly tests whether more sample-efficient image representation can provide the next gain.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
=======
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "detail_kernels",
            torch.tensor(
                [
                    [
                        [
                            [-0.125, 0.0, 0.125],
                            [-0.250, 0.0, 0.250],
                            [-0.125, 0.0, 0.125],
                        ]
                    ],
                    [
                        [
                            [-0.125, -0.250, -0.125],
                            [0.0, 0.0, 0.0],
                            [0.125, 0.250, 0.125],
                        ]
                    ],
                    [
                        [
                            [0.0, 0.250, 0.0],
                            [0.250, -1.0, 0.250],
                            [0.0, 0.250, 0.0],
                        ]
                    ],
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        details = F.conv2d(padded, self.detail_kernels)
        represented = torch.cat((images, details), dim=1)
        return self.classifier(self.features(represented))
>>>>>>> REPLACE