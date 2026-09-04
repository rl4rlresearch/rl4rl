MECHANISM: State-free per-example head normalization

HYPOTHESIS: Replacing the classifier BatchNorm with LayerNorm will exceed 9,311 correct predictions by retaining the successful normalized 147-unit head while removing its dependence on running statistics that are not aligned with tail-averaged parameters.

INTENDED_EDIT: Replace only the dense-head BatchNorm1d with parameter-count-neutral LayerNorm; preserve architecture capacity, training, averaging, augmentation, and weighted-logit TTA.

EVIDENCE: The normalized 147-unit head led to the best architecture and weighted-logit TTA reached 9,311 correct, while reducing tail averaging regressed to 9,307. Attempts to align BatchNorm buffers timed out, leaving state-free head normalization as an untested, computation-light resolution of the same mismatch.

<<<<<<< SEARCH
            nn.BatchNorm1d(147),
=======
            nn.LayerNorm(147),
>>>>>>> REPLACE