MECHANISM: Reference-class logit gauge fixing

HYPOTHESIS: Representing eight-class softmax probabilities with seven learned logits and one fixed-zero reference logit will retain at least 85% validation accuracy while reducing classifier MACs by 119 per example and learned parameters by 120.

INTENDED_EDIT: Change the classifier from eight outputs to seven and append a fixed-zero eighth logit, preserving the full softmax function class and the verified 29-step recurrent architecture.

EVIDENCE: The 60+59 GRU over frames 3–31 met the target at 85.03% accuracy; softmax is invariant to subtracting one class logit from every logit, so fixing that reference logit to zero removes redundant learned computation without reducing representational capacity.

<<<<<<< SEARCH
        self.classifier = nn.Linear(119, 8)
=======
        self.classifier = nn.Linear(119, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden_a, _hidden_b, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        _hidden_a, _hidden_b, summary, count = state
        learned_logits = self.classifier(summary / count.clamp_min(1.0))
        reference_logit = learned_logits.new_zeros((learned_logits.shape[0], 1))
        return torch.cat((learned_logits, reference_logit), dim=1)
>>>>>>> REPLACE