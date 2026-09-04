MECHANISM: Post-global context deferral

HYPOTHESIS: Moving the second-stage 136-token window from the local layer immediately after the first full-context layer to the local layer immediately before final full-context integration will lower val_bpb below 0.983152 without changing attention compute or throughput.

INTENDED_EDIT: Change the six local windows from 144/144/136/136/128/128 to 144/144/136/128/128/136, preserving the proven first-stage taper, 136-token average, and 8-token alignment.

EVIDENCE: The compute-matched 144/144/136/136/128/128 taper is the best result at 0.983152; this permutation isolates window placement within the second three-local-layer stage while keeping its multiset and every other setting unchanged.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            17 * long_window // 256,
            17 * long_window // 256,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            17 * long_window // 256,
            long_window // 16,
            long_window // 16,
            17 * long_window // 256,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 144/144/136/136/128/128 local taper; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 144/144/136/128/128/136 local windows; full context at layers 4 and 8
>>>>>>> REPLACE