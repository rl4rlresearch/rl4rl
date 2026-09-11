MECHANISM: Channel-adaptive mixed downsampling

HYPOTHESIS: Replacing both hard max-pooling stages with learned per-channel mixtures of average and maximum pooling will exceed 9,289 correct predictions by reducing one-pixel phase sensitivity while retaining strong localized activations.

INTENDED_EDIT: Add channel-adaptive mixed pooling at both resolution reductions; remove the redundant affine transform from the classifier’s LayerNorm to remain below the parameter ceiling.

EVIDENCE: Flip-consistency weighting improved correctness from 9,287 to 9,289, indicating sensitivity to transformed evidence. Unlike the failed spatial attention and static spatial pooling designs, this mechanism neither selects absolute locations nor discards channels; it challenges the load-bearing assumption that every feature channel benefits from the same hard maximum downsampling.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
=======
class ChannelAdaptivePool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_weight_logits = nn.Parameter(
            torch.full((channels,), math.log(3.0))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        max_weight = torch.sigmoid(self.max_weight_logits).view(
            1, -1, 1, 1
        )
        return torch.lerp(average, maximum, max_weight)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pool1 = nn.MaxPool2d(2)
=======
        self.pool1 = ChannelAdaptivePool(32)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pool2 = nn.MaxPool2d(2)
=======
        self.pool2 = ChannelAdaptivePool(64)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.LayerNorm(192),
=======
            nn.LayerNorm(192, elementwise_affine=False),
>>>>>>> REPLACE