MECHANISM: More training tokens through tighter local attention

HYPOTHESIS: Reducing short-layer attention from half-context to quarter-context will lower val_bpb below 0.985730 by increasing training throughput while the two full-context layers preserve global token mixing.

INTENDED_EDIT: Change the six short-window layers from 1024-token to 512-token attention; retain the existing SSSL pattern, final full-context layer, optimizer, and schedules.

EVIDENCE: The best design processes 491.0M tokens in five minutes, while six of eight layers use short attention and two retain full attention. Recent weight-decay and warmdown refinements bracketed their local optima without improving 0.985730, motivating an independent compute-efficiency change.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE