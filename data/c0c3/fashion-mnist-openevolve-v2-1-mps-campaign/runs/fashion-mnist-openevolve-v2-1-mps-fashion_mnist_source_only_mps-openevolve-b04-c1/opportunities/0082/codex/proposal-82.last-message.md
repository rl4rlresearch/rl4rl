MECHANISM: Zero-gated local mean-and-contrast refinement

HYPOTHESIS: Adding nonlinear local mean and contrast features with only 128 parameters will exceed 9,328 correct predictions by capturing garment texture and boundary context without the runtime cost that prevented depthwise and dilated spatial refinements from completing.

INTENDED_EDIT: After mirrored-view fusion, add zero-initialized per-channel gates over nonlinear 3×3 local-mean and local-contrast maps, preserving the current classifier exactly at initialization.

EVIDENCE: The current fused classifier reached 9,328 correct, while both depthwise post-fusion refinement and dilated broader-context attempts timed out; this tests the same spatial-context hypothesis with negligible additional computation.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.context_scales = nn.Parameter(torch.zeros(2, 64, 1, 1))
        self.classifier = nn.Sequential(
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
        local_mean = F.avg_pool2d(
            fused, kernel_size=3, stride=1, padding=1
        )
        local_contrast = fused - local_mean
        fused = (
            fused
            + self.context_scales[0] * F.gelu(local_mean)
            + self.context_scales[1] * F.gelu(local_contrast)
        )
        return self.classifier(fused)
>>>>>>> REPLACE