MECHANISM: Learned decision-level canonical-view routing

HYPOTHESIS: Classifying each mirrored view independently and combining its logits with a learned content-dependent gate will exceed 9,328 correct predictions by preserving signed spatial evidence that pre-classifier averaging and absolute disagreement discard.

INTENDED_EDIT: Replace invariant–disagreement feature fusion with a shared nonlinear classifier applied separately to both views, followed by a zero-initialized learned gate that selects their class predictions.

EVIDENCE: The invariant–disagreement model reached 9,316 correct and improved to 9,328 through regularization, while a spatial evidence readout still reached 9,307; this suggests spatially oriented evidence is valuable, but the current load-bearing assumption that mirrored features should be symmetrized before class prediction may erase it.

<<<<<<< SEARCH
        self.view_fusion = nn.Conv2d(
            128, 64, kernel_size=1, bias=False
        )
        with torch.no_grad():
            self.view_fusion.weight.zero_()
            self.view_fusion.weight[:, :64, 0, 0].copy_(torch.eye(64))
=======
        self.view_score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1, bias=False),
        )
        with torch.no_grad():
            self.view_score[1].weight.zero_()
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
        paired_features = torch.cat(
            (features, flipped_features),
            dim=0,
        )
        paired_logits = self.classifier(paired_features)
        paired_scores = self.view_score(paired_features)

        logits, flipped_logits = paired_logits.split(
            batch_size,
            dim=0,
        )
        scores, flipped_scores = paired_scores.split(
            batch_size,
            dim=0,
        )
        view_weights = torch.softmax(
            torch.cat((scores, flipped_scores), dim=1),
            dim=1,
        )
        return (
            view_weights[:, :1] * logits
            + view_weights[:, 1:] * flipped_logits
        )
>>>>>>> REPLACE