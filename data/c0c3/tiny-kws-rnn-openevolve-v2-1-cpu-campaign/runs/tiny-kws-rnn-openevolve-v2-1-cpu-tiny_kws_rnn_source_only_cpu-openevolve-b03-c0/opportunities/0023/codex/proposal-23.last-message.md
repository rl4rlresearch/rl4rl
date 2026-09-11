MECHANISM: Recurrent-coordinate logit substitution

HYPOTHESIS: Replacing one learned classifier row with a fixed, scaled readout of the two unused GRU summary coordinates will retain at least 85% validation accuracy while reducing classifier MACs by 14.3% and preserving seven independent softmax decision dimensions.

INTENDED_EDIT: Reduce the classifier from seven outputs to six, construct a seventh logit from the final two recurrent summary coordinates, and continue deriving the eighth as the negative sum.

EVIDENCE: The seven-output zero-sum head improved accuracy to 86.26% while removing one dense output row, and the current 96-coordinate version still passes at 85.28%; the GRU’s two omitted coordinates provide a no-dense-MAC source for another trainable decision score.

<<<<<<< SEARCH
        self.classifier = nn.Linear(96, 7)
=======
        self.classifier = nn.Linear(96, 6)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.classifier(averaged[:, :96])
        return torch.cat((logits, -logits.sum(dim=1, keepdim=True)), dim=1)
=======
        logits = self.classifier(averaged[:, :96])
        recurrent_logit = 3.0 * averaged[:, 96:98].sum(dim=1, keepdim=True)
        independent_logits = torch.cat((logits, recurrent_logit), dim=1)
        return torch.cat(
            (
                independent_logits,
                -independent_logits.sum(dim=1, keepdim=True),
            ),
            dim=1,
        )
>>>>>>> REPLACE