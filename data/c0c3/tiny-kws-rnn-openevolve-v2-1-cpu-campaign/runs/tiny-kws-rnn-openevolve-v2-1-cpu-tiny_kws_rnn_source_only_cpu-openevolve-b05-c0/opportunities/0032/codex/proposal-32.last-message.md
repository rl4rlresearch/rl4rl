MECHANISM: Reference-class logit parameterization

HYPOTHESIS: Replacing the eight-output classifier with seven learned log-odds and one fixed reference logit will retain at least 85% accuracy while removing 94,540 validation MACs and 117 parameters without reducing softmax expressivity.

INTENDED_EDIT: Preserve the verified 24-step recurrent schedule and emit seven learned logits plus a zero reference-class logit.

EVIDENCE: The current 24-step model achieved 85.28%, while reducing to 23 steps fell to 84.29%; preserving its recurrent computation and compressing the redundant eight-class logit representation is therefore the lower-risk cost reduction.

<<<<<<< SEARCH
        self.classifier = nn.Linear(116, 8)
=======
        self.classifier = nn.Linear(116, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(torch.cat((mean_output, hidden[:, 0, :]), dim=-1))
=======
        relative_logits = self.classifier(
            torch.cat((mean_output, hidden[:, 0, :]), dim=-1)
        )
        reference_logit = torch.zeros_like(relative_logits[:, :1])
        return torch.cat((relative_logits, reference_logit), dim=-1)
>>>>>>> REPLACE