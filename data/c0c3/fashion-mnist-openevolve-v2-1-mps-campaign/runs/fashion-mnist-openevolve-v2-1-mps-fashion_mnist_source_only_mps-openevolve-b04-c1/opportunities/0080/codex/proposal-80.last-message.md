MECHANISM: Shared-head late logit pooling across mirrored views

HYPOTHESIS: Classifying each orientation coherently before averaging its logits will exceed 9,328 correct predictions by preserving asymmetric spatial structure that early invariant/disagreement fusion destroys.

INTENDED_EDIT: Remove hand-designed feature-level mirror fusion and apply the shared nonlinear classifier independently to both views, averaging only their final class logits.

EVIDENCE: The feature-fusion baseline reached 9,328 correct, while the attention-based attempt to replace its static symmetric representation timed out; late logit pooling tests a genuinely different, exact-invariance mechanism while eliminating the 128→64 fusion convolution and avoiding attention’s computational cost.

<<<<<<< SEARCH
        self.pool = nn.MaxPool2d(2)
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _classify_views(
        self,
        features: torch.Tensor,
        flipped_features: torch.Tensor,
    ) -> torch.Tensor:
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
    def _classify_views(
        self,
        features: torch.Tensor,
        flipped_features: torch.Tensor,
    ) -> torch.Tensor:
        paired_features = torch.cat((features, flipped_features), dim=0)
        paired_logits = self.classifier(paired_features)
        logits, flipped_logits = paired_logits.chunk(2, dim=0)
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE