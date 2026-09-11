MECHANISM: Smaller-batch update densification without geometric augmentation

HYPOTHESIS: The verified unaugmented 243,178-parameter CNN trained with batch size 64 and approximately 1,563 optimizer steps will exceed 9,258 correct validation predictions.

INTENDED_EDIT: Remove flip training and flip-ensemble inference while halving batch size, preserving the qualified architecture, loss, optimizer, and exposure-normalized schedule.

EVIDENCE: The unaugmented batch-128 design achieved 9,258 correct, while horizontal flips reduced this to 9,230 and translations to 9,102; batch 64 therefore tests additional optimizer updates without repeating the harmful augmentation changes.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    flip_mask = torch.rand(
        (images.shape[0], 1, 1, 1), device=images.device
    ) < 0.5
    images = torch.where(flip_mask, torch.flip(images, dims=(-1,)), images)
    return images, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE