MECHANISM: Seven-feature nonlinear carry bottleneck

HYPOTHESIS: Reducing the qualified eight-unit GELU MLP to seven units will produce a 972-parameter transformer with at least 99% accuracy, because the eight-unit model achieved 99.78% while preserving both specialized attention heads and the five-dimensional token representation.

INTENDED_EDIT: Reduce `d_ff` from eight to seven, removing 15 learned MLP parameters while leaving the qualified attention, embedding, and output-bias design unchanged.

EVIDENCE: Successive reductions from 10 to nine and then eight MLP units achieved 99.96%, 99.94%, and 99.78% accuracy respectively; testing the adjacent width of seven is the narrowest continuation of this consistently successful compression path.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=8)
=======
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE