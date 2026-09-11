MECHANISM: Evenly spaced global-context mixing

HYPOTHESIS: Moving the three full-context layers from indices 3, 5, and 7 to evenly spaced indices 1, 4, and 7 will reduce val_bpb below 0.984227 by exposing earlier representations to global context while preserving throughput and parameter count.

INTENDED_EDIT: Change only the attention window pattern, retaining three full-context layers, the 78% linear warmdown, batch size, optimizer, and all other architecture settings.

EVIDENCE: Adding a third full-context layer improved val_bpb from 0.984312 to 0.984227 despite reducing processed tokens from 501.6M to 495.5M, showing global mixing is valuable; redistributing the same three layers isolates whether earlier, uniform spacing uses that capacity more effectively.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=quarter context
=======
WINDOW_PATTERN = "SLSSLSSL" # three evenly spaced full-context layers; S=quarter context
>>>>>>> REPLACE