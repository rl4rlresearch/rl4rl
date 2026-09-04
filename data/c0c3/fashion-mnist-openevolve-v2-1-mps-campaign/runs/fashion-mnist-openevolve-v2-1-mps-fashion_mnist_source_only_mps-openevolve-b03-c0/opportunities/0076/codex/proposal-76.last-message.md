MECHANISM: Low-weight boundary-safe translation ensemble

HYPOTHESIS: Mixing 10% reflection-padded one-pixel translation logits with 90% original logits will exceed 9,319 correct predictions by capturing complementary translation signal without allowing shifted views to dominate decisions.

INTENDED_EDIT: During evaluation only, blend the original logits with four reflection-padded one-pixel translations, then apply the existing temperature calibration; training remains unchanged.

EVIDENCE: Equal-weight translation ensembling lost only 19 correct predictions, far less than horizontal-reflection ensembling’s 861-loss, suggesting useful translation signal overwhelmed by excessive shifted-view weight; translation-heavy training likewise fell to 9,289, motivating a conservative residual blend.

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
            shifted_logits = torch.zeros_like(logits)
            for shifted in (
                padded[:, :, 0:28, 1:29],
                padded[:, :, 2:30, 1:29],
                padded[:, :, 1:29, 0:28],
                padded[:, :, 1:29, 2:30],
            ):
                shifted_logits.add_(self.classifier(self.features(shifted)))
            logits = 0.9 * logits + 0.025 * shifted_logits
            logits = logits / 0.738156
        return logits
>>>>>>> REPLACE