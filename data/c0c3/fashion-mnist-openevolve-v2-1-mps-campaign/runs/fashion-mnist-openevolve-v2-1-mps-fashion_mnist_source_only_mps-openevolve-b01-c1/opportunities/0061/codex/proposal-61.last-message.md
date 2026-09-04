MECHANISM: Localized raw-image spatial shortcut

HYPOTHESIS: Supplying pooled intensity and local-maximum maps directly to the successful fixed-coordinate decoder will exceed 9,247 correct predictions by restoring silhouette and texture information lost through two max-pooling stages.

INTENDED_EDIT: Concatenate parameter-free 7×7 average- and max-pooled image maps with the learned 48-channel feature map, expanding only the classifier input and retaining all proven training and ensemble settings.

EVIDENCE: Widening the flattened head regressed to 9,210 and global summaries reached only 9,236, suggesting that generic capacity and spatially collapsed statistics are insufficient; this instead adds localized low-level evidence while preserving the 9,247-correct decoder and curriculum.

<<<<<<< SEARCH
            nn.Linear(48 * 7 * 7, 80),
=======
            nn.Linear(50 * 7 * 7, 80),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        raw_summary = torch.cat(
            (
                F.avg_pool2d(images, kernel_size=4, stride=4),
                F.max_pool2d(images, kernel_size=4, stride=4),
            ),
            dim=1,
        )
        return self.classifier(torch.cat((features, raw_summary), dim=1))
>>>>>>> REPLACE