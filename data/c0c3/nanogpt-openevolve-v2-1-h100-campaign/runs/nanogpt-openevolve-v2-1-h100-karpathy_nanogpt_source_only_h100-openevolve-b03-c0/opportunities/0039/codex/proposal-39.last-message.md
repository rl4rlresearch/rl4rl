MECHANISM: Continued isolated early-boundary context expansion

HYPOTHESIS: Expanding only the first pre-global attention window from 512 to 1,024 tokens will achieve `val_bpb < 0.982068` by strengthening early contextualization while retaining 128-token windows in the other five local layers.

INTENDED_EDIT: Increase the first pre-global window from one-quarter to one-half of the 2,048-token context; preserve architecture, optimizer, batching, schedules, and all other attention windows.

EVIDENCE: Increasing only the first pre-global window from 256 to 512 tokens improved `val_bpb` from `0.982234` to `0.982068` despite slightly lower throughput, directly motivating a power-of-two continuation that isolates whether the early-boundary benefit persists.

<<<<<<< SEARCH
        pre_global_window = long_window // 4
=======
        pre_global_window = long_window // 2
>>>>>>> REPLACE