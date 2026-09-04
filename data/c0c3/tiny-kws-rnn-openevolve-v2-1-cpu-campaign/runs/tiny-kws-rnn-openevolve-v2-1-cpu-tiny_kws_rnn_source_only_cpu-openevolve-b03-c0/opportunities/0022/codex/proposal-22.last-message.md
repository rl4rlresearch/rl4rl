MECHANISM: Zero-sum-head-assisted recurrent readout pruning

HYPOTHESIS: Classifying from 96 recurrent summary coordinates with the seven-output zero-sum head will retain at least 85% validation accuracy while reducing exact classifier MACs; the zero-sum head improved the 97-coordinate model from 85.52% to 86.26%, enough to plausibly recover the cropped 96-coordinate model’s 84.66% result.

INTENDED_EDIT: Reduce the seven-output classifier input width from 97 to 96 and omit one additional recurrent summary coordinate, while preserving the 98-unit GRU and all 32 causal steps.

EVIDENCE: The 96-coordinate conventional head missed narrowly at 84.66%, while the seven-output zero-sum parameterization raised the passing 97-coordinate design to 86.26% and reduced MACs, motivating their combination.

<<<<<<< SEARCH
        self.classifier = nn.Linear(97, 7)
=======
        self.classifier = nn.Linear(96, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.classifier(averaged[:, :97])
=======
        logits = self.classifier(averaged[:, :96])
>>>>>>> REPLACE