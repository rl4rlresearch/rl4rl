MECHANISM: Isolated expanded early local-to-global context

HYPOTHESIS: Expanding only the first pre-global attention window from 256 to 512 tokens will achieve `val_bpb < 0.982234` by improving early contextualization without paying for the harmful late expansion.

INTENDED_EDIT: Use a 512-token window immediately before the first full-context layer while retaining 128-token windows for every other local layer.

EVIDENCE: A 256-token window only at the first boundary achieved the best result (`0.982234`), outperforming both-boundary 256 (`0.982545`); therefore the prior both-boundary 512 result (`0.982674`) does not isolate whether additional early context is beneficial.

<<<<<<< SEARCH
        pre_global_window = long_window // 8
=======
        pre_global_window = long_window // 4
>>>>>>> REPLACE