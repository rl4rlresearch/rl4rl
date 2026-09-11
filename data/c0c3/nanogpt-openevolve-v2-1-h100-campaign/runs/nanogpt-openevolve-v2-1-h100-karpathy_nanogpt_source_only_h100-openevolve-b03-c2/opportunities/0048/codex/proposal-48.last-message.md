MECHANISM: Earlier global-context integration

HYPOTHESIS: Moving the intermediate full-context layer from layer 4 to layer 3 will reduce `val_bpb` below 0.983618 by giving four subsequent local layers access to globally integrated representations.

INTENDED_EDIT: Restore two full-context layers while placing them at layers 3 and 8; retain 384-token local attention and all verified optimizer settings.

EVIDENCE: Removing intermediate global attention regressed from 0.983618 to 0.985116, while early-heavy context allocation nearly matched the best design and late-heavy allocation regressed to 0.984213, motivating earlier global mixing without changing compute.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # 384-token local layers; final layer is forced full-context
=======
WINDOW_PATTERN = "SSLSSSSL" # 384-token local layers; full context at layers 3 and 8
>>>>>>> REPLACE