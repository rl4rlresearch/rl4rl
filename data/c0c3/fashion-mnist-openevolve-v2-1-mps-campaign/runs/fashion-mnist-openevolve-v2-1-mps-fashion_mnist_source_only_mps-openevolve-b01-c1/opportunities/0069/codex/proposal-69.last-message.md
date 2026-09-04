MECHANISM: Dense-head dropout regularization

HYPOTHESIS: Adding 10% dropout to the parameter-dominant fixed-coordinate classifier will exceed 9,247 correct predictions by reducing feature co-adaptation without corrupting the successful input geometry or increasing compute materially.

INTENDED_EDIT: Insert dropout after normalization in the 80-unit classifier head while preserving the proven architecture, augmentation, objectives, optimizer, and evaluation ensemble.

EVIDENCE: Widening the flattened head regressed to 9,210 correct and adding localized raw inputs reached 9,224, suggesting the decoder is not capacity-limited; unlike the 4×4 input cutout that regressed to 9,223, modest head dropout regularizes its roughly 188,000 dense weights without destroying image evidence.

<<<<<<< SEARCH
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
=======
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Dropout(p=0.1),
            nn.Linear(80, 10),
>>>>>>> REPLACE