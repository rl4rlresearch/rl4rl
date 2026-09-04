MECHANISM: Zero-initialized global-average residual classifier

HYPOTHESIS: Adding a lightweight global-average logit branch alongside the verified spatial classifier will exceed 9,328 correct predictions by contributing translation-tolerant distributed-shape evidence without disrupting the baseline function at initialization.

INTENDED_EDIT: Add a zero-initialized 64-to-10 classifier over globally averaged fused features and sum its logits with the existing spatial classifier.

EVIDENCE: Global-max evidence reached 9,325 and dense-head widening reached 9,300, suggesting that neither replacing spatial evidence nor adding generic head capacity is sufficient; a complementary average-pooled residual branch isolates distributed global evidence with negligible computation.

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
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
        self.global_classifier = nn.Linear(64, 10)
        with torch.no_grad():
            self.global_classifier.weight.zero_()
            self.global_classifier.bias.zero_()
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
        global_features = fused.mean(dim=(-2, -1))
        return self.classifier(fused) + self.global_classifier(global_features)
>>>>>>> REPLACE