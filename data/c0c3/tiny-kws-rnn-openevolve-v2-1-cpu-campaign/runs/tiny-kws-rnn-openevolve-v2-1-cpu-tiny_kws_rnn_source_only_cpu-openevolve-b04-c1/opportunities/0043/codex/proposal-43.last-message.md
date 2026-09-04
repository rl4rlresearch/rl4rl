MECHANISM: Redundant final-state readout removal

HYPOTHESIS: Early/late mean pooling plus max pooling will retain at least 85% validation accuracy while removing 552 classifier MACs per example, reducing total validation MACs from 240.85M to approximately 240.40M.

INTENDED_EDIT: Remove the final hidden-state view from the classifier while preserving the temporal-pyramid summaries, max pooling, 69-unit paired GRU, and 26-frame schedule.

EVIDENCE: The 69-unit model passed at 85.89% only after adding separate early/late summaries; unlike the failed global-mean three-view model, this patch preserves that successful temporal structure and removes the final state, which is already represented within the late summary and recurrent maximum.

<<<<<<< SEARCH
        self.classifier = nn.Linear(276, 8)
=======
        self.classifier = nn.Linear(207, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled = torch.cat(
            (
                early_summary / early_count,
                late_summary / late_count,
                maximum,
                hidden[:, 0, :],
            ),
            dim=1,
        )
=======
        pooled = torch.cat(
            (
                early_summary / early_count,
                late_summary / late_count,
                maximum,
            ),
            dim=1,
        )
>>>>>>> REPLACE