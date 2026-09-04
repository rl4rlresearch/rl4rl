MECHANISM: Late decision-level flip symmetrization

HYPOTHESIS: Classifying each orientation before averaging logits will exceed 9,330 correct predictions by preserving complete position-sensitive spatial representations through the nonlinear classifier instead of collapsing them during early coordinatewise fusion.

INTENDED_EDIT: Remove invariant/disagreement feature fusion and apply the shared classifier independently to original and flipped feature maps, averaging their logits for exact horizontal-flip invariance.

EVIDENCE: Global pooling fell to 9,290, demonstrating that spatial layout is important; the current early fusion partially destroys that layout before classification, whereas late symmetrization retains it while cleanly challenging the shared assumption that flip invariance must be constructed at the feature level.

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
        batch_size = features.shape[0]
        view_logits = self.classifier(
            torch.cat((features, flipped_features), dim=0)
        )
        return 0.5 * (
            view_logits[:batch_size] + view_logits[batch_size:]
        )
>>>>>>> REPLACE