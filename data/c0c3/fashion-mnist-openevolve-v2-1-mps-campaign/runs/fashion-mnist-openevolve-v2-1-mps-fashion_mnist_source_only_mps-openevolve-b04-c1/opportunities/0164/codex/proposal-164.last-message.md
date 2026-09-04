MECHANISM: Zero-initialized multiplicative view-agreement fusion

HYPOTHESIS: Adding a local second-order agreement channel while preserving the baseline fusion path at initialization will exceed 9,328 correct predictions by exposing symmetric co-activations that the existing linear 128→64 projection may discard.

INTENDED_EDIT: Expand view fusion to accept the invariant mean, absolute disagreement, and elementwise product of the two orientation features; initialize the new branch to zero so initial behavior remains unchanged.

EVIDENCE: Generic post-fusion nonlinearity scored 9,320 and decision-level fusion scored 9,278, so this patch retains the successful position-sensitive classifier and existing fusion path while adding only a targeted, inexpensive interaction unavailable to the linear projection before channel compression.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
=======
        self.view_fusion = nn.Conv2d(
            192, 64, kernel_size=1, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
=======
        invariant = 0.5 * (features + flipped_features)
        disagreement = torch.abs(features - flipped_features)
        agreement = features * flipped_features
        fused = self.view_fusion(
            torch.cat((invariant, disagreement, agreement), dim=1)
        )
>>>>>>> REPLACE