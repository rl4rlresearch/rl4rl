MECHANISM: Compute-neutral earlier global-context injection

HYPOTHESIS: Moving the first full-context attention layer one block earlier while retaining six half-context and two full-context layers will give global information one additional nonlinear transformation and lower val_bpb below 0.984068 without reducing throughput.

INTENDED_EDIT: Replace the repeated four-layer window pattern with an explicit eight-layer pattern that shifts the first full-context layer from index 3 to index 2 while keeping the final layer full-context.

EVIDENCE: Half-context windows outperformed both 768-token windows at 0.984407 and 1,280-token windows at 0.985631, suggesting the 1,024-token width is already near the useful tradeoff; changing full-context placement while preserving window widths and counts isolates whether earlier global integration is more effective.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSLSSSSL" # two full-context layers, with earlier global integration
>>>>>>> REPLACE