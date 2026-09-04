MECHANISM: Flip-symmetric coordinate-conditioned convolution

HYPOTHESIS: Injecting vertical position and horizontal distance-from-center into the convolutional trunk will exceed 9,249 correct predictions by letting early filters distinguish geometrically different garment regions.

INTENDED_EDIT: Replace the assumption that shared local filters should remain position-agnostic until the flattened head with fixed, flip-symmetric coordinate channels mixed into every first-layer feature.

EVIDENCE: Content-addressed attention pooling fell to 9,228 correct, while widening the fixed-position bottleneck fell to 9,213; this suggests spatial geometry is load-bearing but additional late-head capacity is not, motivating earlier position-conditioned feature extraction.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
=======
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        axis = torch.linspace(-1.0, 1.0, 28)
        vertical_position = axis[:, None].expand(28, 28)
        horizontal_radius = axis.abs()[None, :].expand(28, 28)
        self.register_buffer(
            "coordinate_channels",
            torch.stack((vertical_position, horizontal_radius)).unsqueeze(0),
            persistent=False,
        )
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        coordinates = self.coordinate_channels.expand(
            images.shape[0], -1, -1, -1
        )
        positioned_images = torch.cat((images, coordinates), dim=1)
        return self.classifier(self.features(positioned_images))
>>>>>>> REPLACE