MECHANISM: One-step local diagonal-weight refinement

HYPOTHESIS: A diagonal weight of 0.716137752532958984375 will retain 9,192 correct predictions while reducing validation cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Move the diagonal TTA weight halfway from the best-known setting toward the nearest tested upper setting, preserving exact ensemble normalization.

EVIDENCE: Weight 0.7161376953125 achieved the best observed cross-entropy, while 0.71613780975341796875 was slightly worse with the same correctness; their midpoint is the most informative remaining probe of the local minimum.

<<<<<<< SEARCH
                weight = 0.7161365509033203125 if is_diagonal else 1.0
=======
                weight = 0.716137752532958984375 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.7290924072265625
=======
        return logit_sum / 15.729102020263671875
>>>>>>> REPLACE