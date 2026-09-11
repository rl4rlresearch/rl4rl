MECHANISM: Fine-grained late-global context compression

HYPOTHESIS: A 896-token final-layer window will achieve `val_bpb < 0.981559` by recovering additional throughput without the severe context loss observed at 512 tokens.

INTENDED_EDIT: Reduce only the final attention window from 1,024 to 896 tokens while preserving every other architecture, optimizer, batching, and schedule setting.

EVIDENCE: A 1,024-token final window produced the best result (`0.981559`), while 1,280 tokens slightly regressed to `0.981659` and 512 tokens sharply regressed to `0.985963`; 896 tokens probes the unexplored interval immediately below the current optimum.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window // 2, 0)
=======
        window_sizes[-1] = (long_window * 7 // 16, 0)
>>>>>>> REPLACE