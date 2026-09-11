MECHANISM: Float32-neighbor TTA weight refinement

HYPOTHESIS: The representable float32 diagonal weight immediately below the best-known effective weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Set the diagonal TTA weight to 0.7161376476287841796875 and normalize by its exact total ensemble weight.

EVIDENCE: The best result used 0.7161376953125, while the adjacent upper float32 setting tested via 0.716137752532958984375 was worse; the untested lower float32 neighbor is therefore the most informative remaining local probe.

<<<<<<< SEARCH
                weight = 0.716137409210205078125 if is_diagonal else 1.0
=======
                weight = 0.7161376476287841796875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.729099273681640625
=======
        return logit_sum / 15.7291011810302734375
>>>>>>> REPLACE