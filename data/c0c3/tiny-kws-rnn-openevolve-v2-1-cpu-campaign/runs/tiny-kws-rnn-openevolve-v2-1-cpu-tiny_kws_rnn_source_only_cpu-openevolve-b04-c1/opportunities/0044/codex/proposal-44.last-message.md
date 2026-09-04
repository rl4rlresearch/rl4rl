MECHANISM: Softmax gauge-dimension elimination

HYPOTHESIS: Replacing eight learned logits with seven learned relative logits will preserve at least 85% accuracy while reducing total validation MACs by 168,705, from 240,404,625 to 240,235,920.

INTENDED_EDIT: Produce seven classifier outputs, append a fixed reference logit, and mean-center all eight logits without changing the recurrent model or the representable class probabilities.

EVIDENCE: The current 69-unit paired-GRU design passes at 85.28%. Softmax probabilities have only seven independent logit differences, so removing the unidentifiable common-offset degree reduces classifier cost without reducing predictive expressiveness.

<<<<<<< SEARCH
        self.classifier = nn.Linear(207, 8)
=======
        self.classifier = nn.Linear(207, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(pooled)
=======
        relative_logits = self.classifier(pooled)
        logits = torch.cat(
            (
                relative_logits,
                torch.zeros_like(relative_logits[:, :1]),
            ),
            dim=1,
        )
        return logits - logits.mean(dim=1, keepdim=True)
>>>>>>> REPLACE