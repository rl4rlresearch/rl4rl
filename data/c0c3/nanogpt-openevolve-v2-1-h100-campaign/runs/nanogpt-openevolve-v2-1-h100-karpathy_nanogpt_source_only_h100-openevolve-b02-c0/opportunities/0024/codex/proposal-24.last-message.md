MECHANISM: Stronger front-loaded quadratic Muon regularization

HYPOTHESIS: Increasing initial Muon weight decay from 0.15 to 0.20 while retaining quadratic annealing will reduce val_bpb below 0.985730.

INTENDED_EDIT: Raise the initial Muon weight-decay coefficient to 0.20, preserving its quadratic schedule and all other settings.

EVIDENCE: Raising quadratic-decayed weight decay from 0.125 to 0.15 improved val_bpb from 0.986290 to 0.985730; testing 0.20 determines whether that improvement continues despite 0.20 having been excessive under the higher-cumulative linear schedule.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.15     # continue tuning early regularization under quadratic decay
=======
WEIGHT_DECAY = 0.20     # test stronger early regularization under quadratic decay
>>>>>>> REPLACE