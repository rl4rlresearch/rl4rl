MECHANISM: Two-percent residual translation ensemble

HYPOTHESIS: Blending 2% reflection-padded translation logits will preserve all 9,319 predictions while lowering validation cross-entropy below 0.200074794.

INTENDED_EDIT: During evaluation, blend original logits with the mean logits from four one-pixel translations at 98:2, then apply the existing temperature calibration.

EVIDENCE: A 10% translation blend lowered cross-entropy to 0.199054799 but lost four correct predictions; reducing its influence to 2% should retain part of the complementary signal with substantially less risk of changing argmax decisions.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits / 0.738156
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
            shifted_images = torch.cat(
                (
                    padded[:, :, 1:29, 0:28],
                    padded[:, :, 1:29, 2:30],
                    padded[:, :, 0:28, 1:29],
                    padded[:, :, 2:30, 1:29],
                ),
                dim=0,
            )
            shifted_logits = self.classifier(self.features(shifted_images))
            shifted_logits = shifted_logits.reshape(4, images.shape[0], 10).mean(dim=0)
            logits = (0.98 * logits + 0.02 * shifted_logits) / 0.738156
        return logits
>>>>>>> REPLACE