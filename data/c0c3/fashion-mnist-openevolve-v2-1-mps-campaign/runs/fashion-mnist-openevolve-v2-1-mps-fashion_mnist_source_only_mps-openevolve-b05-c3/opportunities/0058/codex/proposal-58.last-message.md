MECHANISM: Per-channel mixed final pooling

HYPOTHESIS: A learnable, max-biased mixture of max and average pooling in the final stage will exceed 9,172 correct predictions by recovering diffuse local evidence while preserving the validated position-sensitive 3×3 representation.

INTENDED_EDIT: Restore the best verified all-tensor EMA and replace only the final max pool with a lightweight per-channel max/average mixture initialized close to max pooling.

EVIDENCE: The 19/32 curriculum with all-tensor EMA achieved 9,172 correct; the hybrid spatial/global head motivated preserving distributed evidence but timed out, so this 96-parameter pooling change tests that mechanism with negligible computational overhead.

<<<<<<< SEARCH
BASE_LR = 3.0e-3


class ImageClassifier(nn.Module):
=======
BASE_LR = 3.0e-3


class ChannelMixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.full((1, channels, 1, 1), 2.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        max_pooled = F.max_pool2d(images, 2)
        average_pooled = F.avg_pool2d(images, 2)
        max_fraction = self.max_logit.sigmoid()
        return (
            max_fraction * max_pooled
            + (1.0 - max_fraction) * average_pooled
        )


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
=======
            if index in (1, 3):
                layers.append(nn.MaxPool2d(2))
            elif index == 5:
                layers.append(ChannelMixedPool2d(out_channels))
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    ]
=======
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
    ]
>>>>>>> REPLACE