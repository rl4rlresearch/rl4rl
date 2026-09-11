MECHANISM: Earlier-biased endpoint-sampled tail quadrature

HYPOTHESIS: Recency^0.75 weighting will exceed 9,287 correct predictions because correctness declined as the exponent increased from 1.0 to 1.25 and 1.5, suggesting that modestly earlier tail coverage improves decision boundaries.

INTENDED_EDIT: Retain the strongest ten-view logit pooling and half-density quadrature while reducing the parameter-averaging recency exponent from 1.25 to 0.75.

EVIDENCE: Linear-recency weighting achieved 9,287 correct, versus 9,286 for both recency^1.25 and recency^1.5; testing 0.75 probes the accuracy-favoring direction without adding runtime or parameters.

<<<<<<< SEARCH
        recency_power = 1.25
=======
        recency_power = 0.75
>>>>>>> REPLACE