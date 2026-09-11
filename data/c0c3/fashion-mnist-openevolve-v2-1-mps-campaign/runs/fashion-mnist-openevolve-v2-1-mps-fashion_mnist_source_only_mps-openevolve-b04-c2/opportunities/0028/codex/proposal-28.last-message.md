MECHANISM: Longer-horizon batch-normalization statistics without translation augmentation

HYPOTHESIS: Restoring unaugmented training and setting all four batch-normalization momenta to 0.02 will reproduce the verified 9,276-correct result and outperform the current 9,121-correct implementation.

INTENDED_EDIT: Remove the harmful random-translation augmentation while applying momentum 0.02 to every batch-normalization layer; retain the verified architecture, optimizer, schedule, and 1.17817 evaluation calibration.

EVIDENCE: Reference Design 1 achieved 9,276 correct with momentum 0.02 and unaugmented training, whereas the current translation augmentation reduced correctness to 9,121; the previous attempt to restore these settings could not be verified.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128)
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
=======
            nn.BatchNorm2d(32, momentum=0.02),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.02),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.02),
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    padded = F.pad(images, (2, 2, 2, 2))
    crops = padded.unfold(2, 28, 1).unfold(3, 28, 1)
    batch_indices = torch.arange(images.shape[0], device=images.device)
    offsets_y = torch.randint(0, 5, (images.shape[0],), device=images.device)
    offsets_x = torch.randint(0, 5, (images.shape[0],), device=images.device)
    images = crops[
        batch_indices, :, offsets_y, offsets_x
    ].contiguous()
    return images, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE