MECHANISM: Four-point quadratic cautious-decay refinement

HYPOTHESIS: Setting Muon weight decay to 0.1245 will improve upon 0.984068 by targeting the quadratic minimum implied by the verified 0.11, 0.12, 0.126, and 0.14 results.

INTENDED_EDIT: Lower only the cautious Muon weight-decay coefficient from 0.126 to 0.1245, preserving the best 176K-token batch regime and all other settings.

EVIDENCE: Weight decay 0.126 achieved the best val_bpb of 0.984068, versus 0.984109 at 0.12, 0.984418 at 0.14, and 0.984588 at 0.11; fitting this local bracket places the estimated minimum near 0.1245.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.126    # interpolate the bracketed cautious-decay optimum
=======
WEIGHT_DECAY = 0.1245   # refine the four-point cautious-decay optimum
>>>>>>> REPLACE