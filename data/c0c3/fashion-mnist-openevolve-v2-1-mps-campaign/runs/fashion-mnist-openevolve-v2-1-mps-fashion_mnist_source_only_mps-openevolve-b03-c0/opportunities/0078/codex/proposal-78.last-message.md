MECHANISM: Argmax-preserving residual translation ensemble

HYPOTHESIS: Applying the validated 10% translation blend only when it preserves each original argmax will retain all 9,319 correct predictions while lowering validation cross-entropy below 0.200074794.

INTENDED_EDIT: During evaluation, compute the mean logits of four reflection-padded one-pixel translations, form the prior 90:10 blend, and revert per-image blends that would change the original prediction.

EVIDENCE: The unconditional 10% blend lowered cross-entropy from 0.200074794 to 0.199054799 but lost four correct predictions; argmax gating directly removes those prediction changes while retaining the blend on stable examples.

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
            translated_images = torch.cat(
                (
                    padded[:, :, 1:29, 0:28],
                    padded[:, :, 1:29, 2:30],
                    padded[:, :, 0:28, 1:29],
                    padded[:, :, 2:30, 1:29],
                ),
                dim=0,
            )
            translated_logits = self.classifier(
                self.features(translated_images)
            ).reshape(4, images.shape[0], 10).mean(dim=0)
            blended_logits = 0.90 * logits + 0.10 * translated_logits
            preserves_argmax = blended_logits.argmax(dim=1).eq(
                logits.argmax(dim=1)
            )
            logits = torch.where(
                preserves_argmax.unsqueeze(1),
                blended_logits,
                logits,
            )
            logits = logits / 0.738156
        return logits
>>>>>>> REPLACE