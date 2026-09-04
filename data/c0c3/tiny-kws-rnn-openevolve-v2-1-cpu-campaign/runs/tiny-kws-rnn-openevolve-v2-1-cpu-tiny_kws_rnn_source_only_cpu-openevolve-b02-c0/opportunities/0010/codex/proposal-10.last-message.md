MECHANISM: Parameter-free grouped recurrent readout

HYPOTHESIS: Pooling the verified 99-unit recurrent summary into 33 groups before classification will retain at least 85% accuracy while reducing exact classifier MACs and learned parameters.

INTENDED_EDIT: Preserve the 99-unit GRU and 31-step schedule, but average each consecutive group of three recurrent features and replace the 99-to-8 classifier with a 33-to-8 classifier.

EVIDENCE: The 99-unit, 31-step model achieved 85.03% accuracy, while even a 98-unit model using all 32 frames achieved only 84.54%; this motivates preserving recurrent capacity and structurally reducing the readout instead.

<<<<<<< SEARCH
        self.classifier = nn.Linear(99, 8)
=======
        self.classifier = nn.Linear(33, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(summary / count.clamp_min(1.0))
=======
        averaged = summary / count.clamp_min(1.0)
        pooled = averaged.reshape(averaged.shape[0], 33, 3).mean(dim=-1)
        return self.classifier(pooled)
>>>>>>> REPLACE