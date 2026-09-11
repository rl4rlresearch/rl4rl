MECHANISM: Incremental recurrent-readout coordinate pruning

HYPOTHESIS: Reducing the classifier input from 96 to 95 recurrent-summary coordinates will preserve at least 85% validation accuracy while lowering total inference MACs at the successful 25-step schedule.

INTENDED_EDIT: Keep the GRU, feature pooling, and frame schedule unchanged, but remove one classifier input coordinate.

EVIDENCE: The current 25-step design passes at 85.03%, whereas the tested 24-step schedule falls to 84.66%; a one-coordinate readout reduction is a smaller structural change that improves exact MAC cost without further reducing temporal coverage.

<<<<<<< SEARCH
        self.classifier = nn.Linear(96, 7)
=======
        self.classifier = nn.Linear(95, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.classifier(averaged[:, :96])
=======
        logits = self.classifier(averaged[:, :95])
>>>>>>> REPLACE