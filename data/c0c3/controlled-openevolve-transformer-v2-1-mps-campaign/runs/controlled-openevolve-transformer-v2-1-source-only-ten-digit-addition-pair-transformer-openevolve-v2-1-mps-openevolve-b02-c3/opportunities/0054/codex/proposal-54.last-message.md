MECHANISM: Six-feature nonlinear carry bottleneck

HYPOTHESIS: Reducing the qualified seven-unit GELU MLP to six units will produce a 957-parameter transformer with at least 99% accuracy, because the seven-unit model achieved 99.66% while preserving both specialized attention heads and the five-dimensional token representation.

INTENDED_EDIT: Reduce `d_ff` from nine to six, reconstructing the qualified architecture at the next untested width below the verified seven-unit design and removing 45 parameters from the current implementation.

EVIDENCE: Successive widths of ten, nine, eight, and seven achieved 99.96%, 99.94%, 99.78%, and 99.66% accuracy; six units are therefore the narrowest informative continuation of the consistently successful MLP-compression path.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=9)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE