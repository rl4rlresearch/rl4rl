MECHANISM: Reference-class softmax parameterization

HYPOTHESIS: Anchoring one of eight logits at zero will preserve at least 85% accuracy because eight-class softmax has only seven independent logit differences, while reducing classifier MACs by 12.5% and total validation MACs by 221,680.

INTENDED_EDIT: Change the classifier to produce seven learned logits and append a zero reference logit, preserving the successful 68-unit, 23-step recurrent model and four-statistic readout.

EVIDENCE: The current 68-unit boundary-aware model qualifies at 85.52% accuracy; the failed 67-unit pruning shows recurrent capacity should be retained, motivating an expressivity-preserving reduction confined to the classifier.

<<<<<<< SEARCH
        self.classifier = nn.Linear(272, 8)
=======
        self.classifier = nn.Linear(272, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(features)
=======
        relative_logits = self.classifier(features)
        return F.pad(relative_logits, (0, 1))
>>>>>>> REPLACE