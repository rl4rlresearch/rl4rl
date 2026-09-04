MECHANISM: Midpoint refinement of flipped-view fusion plateau

HYPOTHESIS: A 0.508625 flipped-view weight will retain 9,331 correct predictions while lowering cross-entropy below 0.196251647 by reducing asymmetry relative to the best verified 0.50875 fusion.

INTENDED_EDIT: Replace equal evaluation-time fusion with 0.491375 original-view and 0.508625 flipped-view weighting, retaining the verified 1.184 calibration.

EVIDENCE: Flipped weight 0.50875 achieved 9,331 correct with lower cross-entropy than 0.509, while equal fusion achieved only 9,330; 0.508625 conservatively bisects the interval between the best verified point and the unresolved 0.5085 trial.

<<<<<<< SEARCH
            logits = 1.184 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.491375 * logits + 0.508625 * flipped_logits)
>>>>>>> REPLACE