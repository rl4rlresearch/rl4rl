MECHANISM: Channelwise global-salience skip features

HYPOTHESIS: Appending each fused channel’s global maximum to the full spatial representation will exceed 9,328 correct predictions by adding translation-tolerant evidence without distorting inputs or sacrificing spatial detail.

INTENDED_EDIT: Add 64 parameter-free global-max features to the classifier input, increasing learned parameters from 224,442 to 227,514 with negligible additional computation.

EVIDENCE: Two-pixel translation augmentation reduced correct predictions from 9,328 to 9,206, while widening the dense head to 249,618 parameters reached only 9,300; this motivates a compact architectural source of positional robustness rather than more augmentation or generic capacity.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (7 * 7 + 1), 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        global_max = fused.amax(dim=(-2, -1))
        classifier_input = torch.cat(
            (fused.flatten(1), global_max),
            dim=1,
        )
        return self.classifier(classifier_input)
>>>>>>> REPLACE