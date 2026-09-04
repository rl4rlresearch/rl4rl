MECHANISM: Single-channel mean-terminal readout fusion

HYPOTHESIS: Fusing only one of 63 mean/final feature pairs will retain at least 85% validation accuracy while reducing total inference MACs from 270,126,045 to approximately 270,119,525.

INTENDED_EDIT: Reduce the classifier input from 252 to 251 features by averaging one corresponding mean and terminal channel while preserving every other summary feature and the qualified 63-unit, 21-step recurrent path.

EVIDENCE: The 63-unit four-summary model qualified at 85.03%, while fusing all 63 mean/final pairs failed at 83.44%; contracting only one pair is the smallest structural test of whether the broad fusion was too aggressive.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
=======
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(251, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat(
                (mean_output, final_output, maximum, minimum),
                dim=-1,
            )
        )
=======
        fused_channel = 0.5 * (
            mean_output[:, -1:] + final_output[:, -1:]
        )
        return self.classifier(
            torch.cat(
                (
                    mean_output[:, :-1],
                    fused_channel,
                    final_output[:, :-1],
                    maximum,
                    minimum,
                ),
                dim=-1,
            )
        )
>>>>>>> REPLACE