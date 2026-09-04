MECHANISM: Split-aligned batching with center-biased TTA fusion

HYPOTHESIS: Using full batches of 50 will reduce optimizer steps from 2,084 to 2,000 and avoid timeout, while weighting both centered views at 2.25 will change ensemble decisions and exceed 9,286 validation-correct predictions.

INTENDED_EDIT: Increase batch size from 48 to 50, increase both centered-view ensemble weights from 2.0 to 2.25, and normalize by the resulting total weight of 12.5.

EVIDENCE: Three center-weighted TTA attempts timed out without testing the hypothesis, while the verified baseline took 79.5 seconds and 2,084 steps; batch size 50 exactly divides both the 50,000-image split and 100,000-example exposure budget, producing 2,000 full updates.

<<<<<<< SEARCH
BATCH_SIZE = 48
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 50
GRAD_CLIP_NORM = 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
=======
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.25)
>>>>>>> REPLACE

<<<<<<< SEARCH
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
=======
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.25)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.912
=======
        return (ensemble - math.log(12.5)) / 0.912
>>>>>>> REPLACE