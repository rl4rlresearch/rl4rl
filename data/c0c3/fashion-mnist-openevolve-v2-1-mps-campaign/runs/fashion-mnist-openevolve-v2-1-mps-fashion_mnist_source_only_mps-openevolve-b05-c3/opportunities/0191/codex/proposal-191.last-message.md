MECHANISM: Budget-filling classifier bottleneck expansion

HYPOTHESIS: Widening the classifier bottleneck from 48 to 52 units will exceed 9,210 correct predictions by increasing learned class-separation capacity while remaining under the 250,000-parameter ceiling.

INTENDED_EDIT: Expand both classifier linear layers consistently to 52 hidden units, producing 249,318 parameters, and restore the verified 1.178× output calibration.

EVIDENCE: Reference Design 1 reached 9,210 correct with the unchanged 48-unit head, while further TTA tuning preserved or reduced accuracy and specialized multi-scale and channel-recalibration additions regressed; this motivates a minimal capacity increase that preserves the qualified feature extractor and training pipeline.

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
=======
            nn.Linear(96 * 3 * 3, 52),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(52, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
=======
        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE