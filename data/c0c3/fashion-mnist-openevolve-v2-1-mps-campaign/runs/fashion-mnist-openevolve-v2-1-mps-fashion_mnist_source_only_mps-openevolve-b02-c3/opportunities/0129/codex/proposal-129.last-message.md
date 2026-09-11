MECHANISM: One-ULP reverse ensemble temperature calibration

HYPOTHESIS: Raising the positive temperature by one float32 ULP will preserve all 9,348 argmax predictions while lowering validation cross-entropy below 0.18770656127929689.

INTENDED_EDIT: Keep the best verified crop weights and change the temperature from float32 0.753170013427734375 to its immediately higher representable value.

EVIDENCE: The current design has the lowest verified cross-entropy, while lowering its temperature by one ULP worsened cross-entropy without changing correctness; the symmetric higher-temperature probe is therefore the most informative remaining local calibration test.

<<<<<<< SEARCH
        ).log() / 0.75317
=======
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE