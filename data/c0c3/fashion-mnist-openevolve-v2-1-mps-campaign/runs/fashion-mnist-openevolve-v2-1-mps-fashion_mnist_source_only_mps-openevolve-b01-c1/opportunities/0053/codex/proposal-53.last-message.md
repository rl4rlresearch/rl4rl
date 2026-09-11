MECHANISM: Cosine-ramped flip-pair supervision

HYPOTHESIS: Ramping flip-pair allocation from 25% early to 50% late will exceed 9,246 correct predictions by preserving individual-view feature learning early and emphasizing validation-aligned flip aggregation near convergence.

INTENDED_EDIT: Replace the fixed 37.5% flip-pair allocation with a cosine curriculum having the same mean allocation, while leaving the architecture, optimizer, augmentation, and ensemble-loss schedule unchanged.

EVIDENCE: Fixed flip-pair supervision improved correctness from 9,239 at 0% to 9,245 at 25% and 9,246 at 37.5%, while 43.75% regressed to 9,242; a temporal allocation tests whether stronger supervision is useful specifically after individual-view features have formed.

<<<<<<< SEARCH
    pair_loss = 0.9 * full_pair_loss + 0.1 * central_pair_loss
    view_loss = 0.625 * individual_loss + 0.375 * pair_loss
=======
    pair_loss = 0.9 * full_pair_loss + 0.1 * central_pair_loss
    pair_weight = 0.375 - 0.125 * math.cos(math.pi * progress)
    view_loss = (
        (1.0 - pair_weight) * individual_loss + pair_weight * pair_loss
    )
>>>>>>> REPLACE