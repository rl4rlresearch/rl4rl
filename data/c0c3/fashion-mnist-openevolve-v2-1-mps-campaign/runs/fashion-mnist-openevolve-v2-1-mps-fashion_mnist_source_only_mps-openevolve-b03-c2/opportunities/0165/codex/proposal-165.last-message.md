MECHANISM: Single-ULP downward logit-temperature refinement

HYPOTHESIS: The next float32 logit scale below 1.35603 will preserve exactly 9,257 correct predictions and reduce validation cross-entropy below 0.21441890335083008.

INTENDED_EDIT: Decrease only the accuracy-invariant final logit scale to the immediately preceding float32 value.

EVIDENCE: Scale 1.35603 has the lowest verified cross-entropy, while 1.3560271 and 1.35602 were worse; probing its immediate lower float32 neighbor is the finest unresolved search within that bracket.

<<<<<<< SEARCH
        return 1.35603 * aggregate_logits
=======
        return 1.3560298681259155 * aggregate_logits
>>>>>>> REPLACE