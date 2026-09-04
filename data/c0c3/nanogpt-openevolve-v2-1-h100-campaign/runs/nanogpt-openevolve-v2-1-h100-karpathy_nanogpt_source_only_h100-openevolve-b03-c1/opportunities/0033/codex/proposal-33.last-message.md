MECHANISM: Dense late global-context refinement

HYPOTHESIS: Promoting layer 6 to full-context attention will reduce val_bpb below 0.984227 by letting every layer after the proven layer-5 global mixer refine globally integrated representations.

INTENDED_EDIT: Preserve the winning three global layers and all training settings, while changing the final local attention layer at index 6 to full context.

EVIDENCE: Adding a third global layer improved val_bpb from 0.984312 to 0.984227 despite lower throughput, whereas widening every local window regressed to 0.985663; this favors concentrated full-context capacity over distributed medium-context compute.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=quarter context
=======
WINDOW_PATTERN = "SSSLSLLL" # four full-context layers, dense in the final three blocks
>>>>>>> REPLACE