MECHANISM: Cross-projection sharing of the softmax-invariant key bias

HYPOTHESIS: Reusing the mean learned query bias across all key-bias coordinates will produce a 1636-parameter model with at least 99% accuracy.

INTENDED_EDIT: Replace the independent shared key-bias scalar with the mean of the learned query bias, while retaining one fused QKV parameter and a symmetric key bias.

EVIDENCE: Symmetric complete key-bias sharing achieved 99.97% at 1637 parameters, whereas fixed-zero key-bias elimination failed; reusing a learned query-bias statistic removes the last independent key parameter without reintroducing a fixed-zero anchor.

<<<<<<< SEARCH
        # Remove four softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 4))
=======
        # Reuse the mean learned query bias across every softmax-invariant
        # key-bias coordinate, leaving only query and value bias parameters.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1].expand(4),
                self.qkv.bias[d_model + 1 : 2 * d_model - 4],
                self.qkv.bias[2 * d_model - 4 :],
            )
        )
=======
        query_bias = self.qkv.bias[:d_model]
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                self.qkv.bias[d_model:],
            )
        )
>>>>>>> REPLACE