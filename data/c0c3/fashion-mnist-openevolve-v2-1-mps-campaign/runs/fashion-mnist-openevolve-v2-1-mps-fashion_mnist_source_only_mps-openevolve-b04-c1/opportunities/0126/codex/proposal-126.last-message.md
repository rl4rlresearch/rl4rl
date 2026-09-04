MECHANISM: Fixed invariant skip with learned residual view fusion

HYPOTHESIS: Preserving the mirrored-feature average as an immutable identity path while learning only a residual correction will exceed 9,328 correct predictions by preventing optimization from degrading the strongest view representation.

INTENDED_EDIT: Zero-initialize the existing fusion convolution and add its output residually to the invariant features, preserving the baseline function and parameter count at initialization.

EVIDENCE: Disagreement-only fusion fell to 9,290 correct while the full invariant pathway reached 9,328, showing that invariant features are essential; a fixed skip protects them while retaining learnable disagreement interactions without extra runtime.

<<<<<<< SEARCH
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
=======
        with torch.no_grad():
            self.view_fusion.weight.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        residual = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(invariant + residual)
>>>>>>> REPLACE