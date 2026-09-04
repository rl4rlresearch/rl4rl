MECHANISM: Conservative directional vertical-shift TTA extrapolation

HYPOTHESIS: Extending the successful vertical asymmetry by 1/4096 will retain 9,348 correct predictions while lowering cross-entropy below 0.1877068977355957.

INTENDED_EDIT: Restore the best verified total vertical and horizontal weights, then transfer an additional 1/4096 weight between the two vertical crops while preserving total ensemble weight.

EVIDENCE: Reference Design 2’s 1/1024 directional transfer retained 9,348 correct and achieved the best verified cross-entropy; a larger additional 1/2048 transfer timed out, so a smaller extrapolation is the most informative low-risk probe.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.31640625,
            1.31640625,
            0.68359375,
            0.68359375,
        )
=======
        crop_weights = (
            3.0,
            1.312255859375,
            1.314697265625,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE