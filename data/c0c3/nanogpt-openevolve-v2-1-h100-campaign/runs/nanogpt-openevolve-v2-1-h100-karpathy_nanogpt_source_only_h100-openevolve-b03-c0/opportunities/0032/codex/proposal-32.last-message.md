MECHANISM: Further local-attention compute reallocation

HYPOTHESIS: Reducing the six local-attention layers from 256-token to 128-token windows while retaining two full-context layers will increase training throughput enough to achieve `val_bpb < 0.982793`.

INTENDED_EDIT: Change `S` layers from one-eighth-context to one-sixteenth-context attention without altering model capacity, optimizer, batch size, schedule, softcap, or full-context layers.

EVIDENCE: Shortening local windows from 1,024 to 512 tokens improved `val_bpb` from `0.984431` to `0.983023`, and shortening them again to 256 tokens improved it to `0.982793` while increasing throughput from 509.1M to 516.6M tokens; this directly tests whether that compute-quality trend continues.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 16
>>>>>>> REPLACE