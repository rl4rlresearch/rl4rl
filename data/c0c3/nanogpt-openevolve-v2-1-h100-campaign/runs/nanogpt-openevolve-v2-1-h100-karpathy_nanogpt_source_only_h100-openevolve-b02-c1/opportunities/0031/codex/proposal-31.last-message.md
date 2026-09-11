MECHANISM: Quadratically interpolated cautious Muon decay

HYPOTHESIS: Setting Muon weight decay to 0.126 will better approximate the optimum bracketed by the 0.11, 0.12, and 0.14 results, lowering val_bpb below 0.984109.

INTENDED_EDIT: Increase only the cautious Muon weight-decay coefficient from 0.12 to 0.126, preserving the best 176K-token batch regime and all other settings.

EVIDENCE: Weight decay 0.12 achieved 0.984109, outperforming both 0.14 at 0.984418 and 0.11 at 0.984588; quadratic interpolation of these three observations places the estimated minimum near 0.126.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.12     # refine update-count-normalized cautious decay
=======
WEIGHT_DECAY = 0.126    # interpolate the bracketed cautious-decay optimum
>>>>>>> REPLACE