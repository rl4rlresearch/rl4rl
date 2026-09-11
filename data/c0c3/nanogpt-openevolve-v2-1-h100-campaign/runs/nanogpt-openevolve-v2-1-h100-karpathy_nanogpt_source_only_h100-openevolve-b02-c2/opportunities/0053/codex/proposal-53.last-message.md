MECHANISM: Fine-grained delayed post-global context allocation

HYPOTHESIS: A six-token layer-4-to-layer-5 transfer will lower val_bpb below 0.982455 by targeting the narrow optimum suggested by the improving eight-token transfer and regressing twelve-token transfer while preserving aggregate attention compute.

INTENDED_EDIT: Use 144/144/130/142/128/128 local windows with full-context attention at layers 3 and 8.

EVIDENCE: Transfers of 0, 8, and 12 tokens produced val_bpb values of 0.982662, 0.982455, and 0.982730; testing six tokens refines the promising interval immediately below the best verified allocation.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            long_window // 16,
            9 * long_window // 128,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            65 * long_window // 1024,
            71 * long_window // 1024,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144/128/144/128/128 local windows; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/130/142/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE