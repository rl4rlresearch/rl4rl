MECHANISM: Increased update density for minimally compressed readout

HYPOTHESIS: Reducing the batch size to 96 will provide enough additional optimizer updates for the single-channel mean/final fusion to recover at least 85% validation accuracy, while retaining 21 recurrent steps and reducing total inference MACs to approximately 270,119,525.

INTENDED_EDIT: Fuse one corresponding mean/final summary channel, reduce the classifier input from 252 to 251 features, and lower training batch size from 128 to 96 without changing the qualified recurrent path or inference schedule.

EVIDENCE: The full 63-unit model qualified at 85.03%, while the otherwise identical one-channel fusion reached 84.79% and was only two validation predictions short; this was a much smaller accuracy deficit than the 62-unit or spectral contractions, motivating a training-only recovery attempt for the smallest known structural reduction.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 96
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Linear(252, 8)
=======
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
            mean_output[:, :1] + final_output[:, :1]
        )
        return self.classifier(
            torch.cat(
                (
                    fused_channel,
                    mean_output[:, 1:],
                    final_output[:, 1:],
                    maximum,
                    minimum,
                ),
                dim=-1,
            )
        )
>>>>>>> REPLACE