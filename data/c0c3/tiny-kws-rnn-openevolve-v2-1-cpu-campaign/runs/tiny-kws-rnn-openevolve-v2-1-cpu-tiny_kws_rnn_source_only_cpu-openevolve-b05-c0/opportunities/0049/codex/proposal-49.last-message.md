MECHANISM: Reference-class logit parameterization

HYPOTHESIS: Replacing the eight-output classifier with seven learned relative logits plus one fixed zero reference logit will retain at least 85% accuracy while saving 114 MACs per validation example and 115 parameters, because softmax probabilities are invariant to subtracting one class logit from all logits.

INTENDED_EDIT: Change the 114-to-8 classifier to 114-to-7 and append a constant eighth logit, preserving the full eight-class softmax function family without altering recurrence or temporal summaries.

EVIDENCE: The verified 114-feature design achieved 85.03%, while further summary compression failed; this targets classifier redundancy instead of discarding recurrent information.

<<<<<<< SEARCH
        self.classifier = nn.Linear(114, 8)
=======
        self.classifier = nn.Linear(114, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
=======
        relative_logits = self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
        reference_logit = relative_logits.new_zeros((relative_logits.shape[0], 1))
        return torch.cat((relative_logits, reference_logit), dim=-1)
>>>>>>> REPLACE