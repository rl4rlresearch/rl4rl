MECHANISM: One-feature recurrent readout trimming

HYPOTHESIS: Keeping the qualified 128-unit, 27-step recurrent path while reducing only its classifier input to 127 features will retain at least 85% validation accuracy and lower exact dense inference MACs.

INTENDED_EDIT: Preserve all recurrent capacity and temporal coverage, but classify from the first 127 pooled recurrent features using a structurally smaller linear layer.

EVIDENCE: The current 128-unit, 27-step model achieved 85.15%, and a fully reduced 127-unit model achieved 85.28% at 28 steps; trimming only one readout feature is a more conservative reduction than shrinking the recurrent state.

<<<<<<< SEARCH
        self.classifier = nn.Linear(128, 8)
=======
        self.classifier = nn.Linear(127, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(summary / count.clamp_min(1.0))
=======
        return self.classifier(summary[:, :127] / count.clamp_min(1.0))
>>>>>>> REPLACE