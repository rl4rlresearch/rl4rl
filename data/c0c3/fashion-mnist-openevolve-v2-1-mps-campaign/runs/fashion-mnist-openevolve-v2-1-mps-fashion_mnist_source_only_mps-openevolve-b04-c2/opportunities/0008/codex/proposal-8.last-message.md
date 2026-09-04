MECHANISM: Pairwise 5×5 grouped spatial refinement

HYPOTHESIS: An unaugmented batch-64 model with 5×5 pairwise grouped refinement will exceed Reference Design 3’s 9,258 correct predictions while remaining below 250,000 parameters.

INTENDED_EDIT: Restore the best verified batch size and unaugmented training path, then replace the 3×3 depthwise layer with a 5×5 groups-64 convolution, retaining the 128-wide classifier and using approximately 248,426 parameters.

EVIDENCE: Reference Design 3 achieved 9,258 correct and the best cross-entropy; translations and flips reduced accuracy, while pointwise mixing with a narrower classifier reached only 9,246. This motivates preserving the verified classifier and spending remaining capacity on coupled spatial refinement.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(
                128, 128, kernel_size=3, padding=1, groups=128, bias=False
            ),
=======
            nn.Conv2d(
                128, 128, kernel_size=5, padding=2, groups=64, bias=False
            ),
>>>>>>> REPLACE

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    padding = 2
    padded = F.pad(
        images, (padding, padding, padding, padding), mode="replicate"
    )
    batch, channels, height, width = images.shape
    padded_width = width + 2 * padding
    offset_y = torch.randint(
        0, 2 * padding + 1, (batch, 1, 1), device=images.device
    )
    offset_x = torch.randint(
        0, 2 * padding + 1, (batch, 1, 1), device=images.device
    )
    rows = torch.arange(height, device=images.device).view(1, height, 1)
    columns = torch.arange(width, device=images.device).view(1, 1, width)
    indices = (rows + offset_y) * padded_width + columns + offset_x
    indices = indices.reshape(batch, 1, height * width).expand(
        -1, channels, -1
    )
    images = torch.gather(padded.flatten(2), 2, indices).reshape_as(images)
    return images, labels
=======
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
>>>>>>> REPLACE