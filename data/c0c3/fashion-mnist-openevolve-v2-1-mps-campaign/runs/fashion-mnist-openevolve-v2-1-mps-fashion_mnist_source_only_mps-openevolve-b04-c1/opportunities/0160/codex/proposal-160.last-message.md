MECHANISM: Decision-level flip ensemble with nonlinear per-view classification

HYPOTHESIS: Classifying each orientation independently before averaging logits will exceed 9,328 correct predictions by preserving coherent position-sensitive spatial configurations through the nonlinear classifier while retaining exact horizontal-flip invariance.

INTENDED_EDIT: Remove feature-level invariant/disagreement fusion and apply the shared classifier separately to original and flipped feature maps, averaging their logits.

EVIDENCE: Global pooling fell to 9,290, showing spatial layout is load-bearing, while dense-head widening reached only 9,300, arguing against capacity as the main limitation. The current fusion mixes mirrored spatial locations before nonlinear classification; decision-level fusion tests whether preserving each view’s layout resolves that representational bottleneck.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        logits = self.classifier(features)
        flipped_logits = self.classifier(flipped_features)
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE