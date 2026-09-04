MECHANISM: Complementary mean-and-final recurrent readout

HYPOTHESIS: Concatenating the final GRU state with the temporal mean will recover the 0.46-point accuracy deficit of the previously tested 25-step schedule, reaching at least 85% accuracy while reducing total inference MACs by approximately 3.8% versus the passing 26-step model.

INTENDED_EDIT: Remove frame 29 from the passing schedule and expand the classifier input to jointly use the mean recurrent output and final hidden state.

EVIDENCE: The same 25-step frame ablation reached 84.540%, narrowly below the target; its mean-only classifier discarded the complementary final recurrent representation, while the passing 26-step model establishes that the retained temporal coverage is near the accuracy boundary.

<<<<<<< SEARCH
        self.classifier = nn.Linear(128, 8)
=======
        self.classifier = nn.Linear(256, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        temporal_mean = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((temporal_mean, hidden[:, 0, :]), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return schedule[1:-2] + schedule[-1:] if len(schedule) > 3 else schedule
=======
        return schedule[1:-3] + schedule[-1:] if len(schedule) > 4 else schedule
>>>>>>> REPLACE