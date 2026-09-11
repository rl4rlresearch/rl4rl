MECHANISM: Stacked local receptive field with a single global readout

HYPOTHESIS: Keeping the kernel-efficient 512-token local windows while removing the intermediate full-context layer will raise token throughput enough to reduce val_bpb below 0.993365; seven stacked local layers can propagate information across the 2048-token sequence before the final global layer.

INTENDED_EDIT: Change the attention pattern from six local/two global layers to seven local/one final global layer.

EVIDENCE: The 512-token design improved val_bpb to 0.993365 with 512.2M tokens, whereas shrinking windows to 256 caused a throughput cliff and val_bpb 1.013136; this motivates retaining 512-token windows and testing a less costly reduction in global-attention frequency.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # seven 512-token local layers; final layer is forced full-context
>>>>>>> REPLACE