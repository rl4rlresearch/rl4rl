MECHANISM: Second value-bias/output-bias redundancy

HYPOTHESIS: Removing a second value-projection bias coordinate will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because the resulting token-independent attention output remains representable by the centered output-projection bias up to a LayerNorm-null common offset.

INTENDED_EDIT: Store all eight query biases but only six value biases, reconstructing the final two value-bias coordinates as zero without changing initialization or causal attention.

EVIDENCE: Removing the first value-bias coordinate achieved 99.92% accuracy at 1619 parameters, while output-projection weight gauges failed; extending the successful bias redundancy by one coordinate is the smallest evidence-backed reduction.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_bias = torch.cat(
            (self.qv_bias[d_model:], self.qv_bias.new_zeros(1))
        )
=======
        v_bias = torch.cat(
            (self.qv_bias[d_model:], self.qv_bias.new_zeros(2))
        )
>>>>>>> REPLACE