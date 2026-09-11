MECHANISM: Reference-class softmax head

HYPOTHESIS: Replacing the eight learned logits with seven learned relative logits and one fixed reference logit will retain at least 85% accuracy while eliminating 100 classifier MACs per example and 101 parameters, because softmax is invariant to a shared logit offset.

INTENDED_EDIT: Change the classifier to produce seven logits and append a fixed zero as the eighth class logit.

EVIDENCE: The 100-unit full-rate GRU barely cleared the requirement at 85.03%, making further recurrent reductions risky; preserving its recurrent computation while removing the mathematically redundant eighth affine score is the lowest-risk structural cost reduction.

<<<<<<< SEARCH
        self.classifier = nn.Linear(100, 8)
=======
        self.classifier = nn.Linear(100, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(summary / count.clamp_min(1.0))
=======
        relative_logits = self.classifier(summary / count.clamp_min(1.0))
        return F.pad(relative_logits, (0, 1))
>>>>>>> REPLACE