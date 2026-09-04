MECHANISM: Softmax reference-class parameterization

HYPOTHESIS: Replacing the 8-output classifier with seven learned logits plus one fixed reference logit will retain at least 85% validation accuracy because softmax is invariant to a shared logit offset, while reducing exact classifier MACs by 12.5% and learned parameters by 100.

INTENDED_EDIT: Change the 99-to-8 classifier to 99-to-7 and append a zero reference-class logit during classification.

EVIDENCE: The current 18-feature, 30-step model meets the requirement at 85.03%; unlike prior hidden-width, readout-pooling, and frame-removal changes, this preserves the full softmax decision family and recurrent computation while structurally reducing dense inference cost.

<<<<<<< SEARCH
        self.classifier = nn.Linear(99, 8)
=======
        self.classifier = nn.Linear(99, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        _hidden, summary, count = state
        learned_logits = self.classifier(summary / count.clamp_min(1.0))
        reference_logit = learned_logits.new_zeros(learned_logits.shape[0], 1)
        return torch.cat((learned_logits, reference_logit), dim=-1)
>>>>>>> REPLACE