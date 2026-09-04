MECHANISM: Post-fusion depthwise spatial refinement

HYPOTHESIS: A zero-initialized depthwise residual convolution over the fused 7×7 map will exceed 9,328 correct predictions by adding local nonlinear context at negligible cost while preserving the successful model at initialization.

INTENDED_EDIT: Add a 3×3 depthwise residual refinement after mirrored-view fusion, increasing learned parameters by 576 to 225,018.

EVIDENCE: The nonlinear spatially fused classifier reached 9,328 correct, while a full low-resolution residual block timed out; refining only the already-fused batch with a depthwise convolution tests the same spatial-context idea with substantially less computation.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        self.fusion_refine = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1,
            groups=64,
            bias=False,
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
            self.fusion_refine.weight.zero_()
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
        fused = fused + F.gelu(self.fusion_refine(fused))
        return self.classifier(fused)
>>>>>>> REPLACE