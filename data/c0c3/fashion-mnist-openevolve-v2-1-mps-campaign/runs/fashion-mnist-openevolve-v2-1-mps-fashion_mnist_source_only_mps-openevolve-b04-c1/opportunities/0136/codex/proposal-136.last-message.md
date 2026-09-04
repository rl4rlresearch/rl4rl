MECHANISM: Near-identity post-fusion spatial refinement

HYPOTHESIS: A lightweight residual bottleneck over the fused 7×7 feature map will exceed 9,328 correct predictions by learning local part relationships that classifier widening and global pooling cannot represent, while its 0.1-scaled initialization preserves the verified pathway.

INTENDED_EDIT: Add a 13,568-parameter convolutional bottleneck after view fusion, initialized as a small residual correction; total learned parameters become 238,010.

EVIDENCE: Dense-head widening reached only 9,300 and a global-average branch reached 9,290, indicating remaining capacity is better allocated to spatial feature processing while retaining the successful fusion and classifier.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.classifier = nn.Sequential(
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
        )
        nn.init.constant_(self.spatial_refine[-1].weight, 0.1)
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
        fused = fused + self.spatial_refine(fused)
        return self.classifier(fused)
>>>>>>> REPLACE