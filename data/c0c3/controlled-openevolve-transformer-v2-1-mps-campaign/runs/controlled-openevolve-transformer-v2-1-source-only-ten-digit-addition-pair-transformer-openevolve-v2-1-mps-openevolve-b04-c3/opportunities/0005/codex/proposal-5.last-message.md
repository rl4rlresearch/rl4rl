MECHANISM: Fused reconstruction of a softmax-invariant key bias

HYPOTHESIS: Reconstructing a zero key-bias slice inside the fused linear operation will preserve baseline numerical behavior and achieve at least 99% accuracy with 1,636 parameters.

INTENDED_EDIT: Store only the learned query and value bias slices while passing a reconstructed full-width bias directly to `F.linear`.

EVIDENCE: The 1,644-parameter baseline reached 99.96%. Earlier key-bias removal preserved initialization but reached 80.83% when biases were applied explicitly; retaining the original fused bias computation tests whether that numerical change caused the regression.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # The key bias is softmax-invariant. Keep only query and value
        # biases in the original Linear parameter slot so initialization
        # and optimizer parameter ordering remain otherwise unchanged.
        self.qkv.bias = nn.Parameter(torch.zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qv_bias = self.qkv.bias
        fused_bias = torch.cat(
            (qv_bias[:d_model], qv_bias.new_zeros(d_model), qv_bias[d_model:])
        )
        qkv = F.linear(x, self.qkv.weight, fused_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE