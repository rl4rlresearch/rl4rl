MECHANISM: Incremental recurrent readout pruning

HYPOTHESIS: Classifying from 96 of the 98 recurrent summary coordinates will retain at least 85% validation accuracy while reducing exact classifier MACs by another 6,520 and learned parameters by 8.

INTENDED_EDIT: Reduce the classifier input width from 97 to 96 and omit one additional summary coordinate, while preserving the verified 98-unit recurrent dynamics and all 32 causal steps.

EVIDENCE: The 97-coordinate readout achieved 85.52% accuracy—higher than the 98-coordinate baseline’s 85.03%—while reducing MACs, indicating that readout pruning is better tolerated than shrinking the recurrent state, spectral input, or temporal coverage.

<<<<<<< SEARCH
        self.classifier = nn.Linear(97, 8)
=======
        self.classifier = nn.Linear(96, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(averaged[:, :97])
=======
        return self.classifier(averaged[:, :96])
>>>>>>> REPLACE