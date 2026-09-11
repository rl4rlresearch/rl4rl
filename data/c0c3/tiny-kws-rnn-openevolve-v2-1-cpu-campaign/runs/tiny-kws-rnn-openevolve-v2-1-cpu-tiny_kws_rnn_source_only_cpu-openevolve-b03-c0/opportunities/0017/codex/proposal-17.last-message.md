MECHANISM: One-coordinate recurrent readout pruning

HYPOTHESIS: Retaining the verified 98-unit recurrent dynamics while excluding one summary coordinate from the classifier will preserve at least 85% validation accuracy and reduce exact classifier MACs by 6,520 over the validation set.

INTENDED_EDIT: Reduce only the classification head input from 98 to 97 and classify from the first 97 coordinates of the averaged recurrent summary.

EVIDENCE: The 98-unit GRU passed at 85.03%, while shrinking the recurrent state to 97 failed; pruning one coordinate only at readout preserves the full recurrent capacity and is a lower-risk structural reduction than altering recurrence, temporal coverage, or spectral inputs.

<<<<<<< SEARCH
        self.classifier = nn.Linear(98, 8)
=======
        self.classifier = nn.Linear(97, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        _hidden, summary, count = state
        averaged = summary / count.clamp_min(1.0)
        return self.classifier(averaged[:, :97])
>>>>>>> REPLACE