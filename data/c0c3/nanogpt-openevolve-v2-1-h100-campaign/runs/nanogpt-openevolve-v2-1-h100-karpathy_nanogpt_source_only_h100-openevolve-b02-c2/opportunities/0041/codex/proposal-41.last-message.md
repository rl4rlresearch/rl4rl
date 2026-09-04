MECHANISM: Penultimate global-context integration

HYPOTHESIS: Moving the second full-context layer from layer 8 to layer 7 while retaining full context at layer 3 and the verified local-window multiset will lower val_bpb below 0.982662 by allowing the final block to transform globally mixed representations without changing aggregate attention compute.

INTENDED_EDIT: Place full-context attention at layers 3 and 7, with 144/144/136/136/128/128 windows across the remaining layers, and remove the unconditional final-layer override.

EVIDENCE: Advancing the first full-context layer from layer 4 to layer 3 improved val_bpb from 0.983152 to 0.982662 at unchanged compute; shifting the second full-context layer one step earlier tests the same benefit while preserving the best first-global placement, whereas the large regression at layer 2 cautions against a larger shift.

<<<<<<< SEARCH
        assert short_idx == len(short_windows)
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
        assert short_idx == len(short_windows)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144/136/136/128/128 local taper; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSLS" # 144/144/136/136/128/128 local taper; full context at layers 3 and 7
>>>>>>> REPLACE