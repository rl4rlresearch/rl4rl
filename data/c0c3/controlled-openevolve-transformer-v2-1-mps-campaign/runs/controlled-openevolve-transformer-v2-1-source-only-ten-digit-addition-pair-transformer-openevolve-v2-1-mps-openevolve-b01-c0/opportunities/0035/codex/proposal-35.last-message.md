MECHANISM: Value-bias/output-bias redundancy

HYPOTHESIS: Removing one value-projection bias coordinate will reduce the model from 1620 to 1619 parameters while retaining at least 99% accuracy, because any resulting token-independent attention output can be represented by the centered output-projection bias up to a LayerNorm-null common offset.

INTENDED_EDIT: Store all eight query biases but only seven value biases, reconstructing the final value bias as zero without changing initialization or the causal attention computation.

EVIDENCE: Centering the attention output-projection bias passed at 99.89%, while modifying output-projection weight columns failed; removing a redundant value-bias coordinate uses the successful bias path and avoids perturbing learned projection weights.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qv_bias.chunk(2)
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        q_bias = self.qv_bias[:d_model]
        v_bias = torch.cat(
            (self.qv_bias[d_model:], self.qv_bias.new_zeros(1))
        )
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
>>>>>>> REPLACE