MECHANISM: Nonzero query-pivot gauge normalization

HYPOTHESIS: Fixing the sole nonzero pivot in the first query row at 0.02 will preserve at least 99% accuracy while reducing the verified 1,139-parameter model to 1,138 learned parameters, because reciprocal scaling of the corresponding shared-key coordinate leaves attention scores representationally unchanged.

INTENDED_EDIT: Remove the learned query pivot scalar, reconstruct it as a fixed nonzero coefficient, and preserve full constructor RNG consumption by discarding its original initialized value.

EVIDENCE: The current 1,139-parameter design achieved 100% accuracy with seven leading query coefficients already fixed. Unlike the timed-out third output-projection constraint and failed rank-reducing value constraint, normalizing the remaining query-row pivot exploits the complementary query/key scale freedom without zeroing that row or reducing attention rank.

<<<<<<< SEARCH
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates and
        # one trailing value coefficient are fixed. Constructing the original
        # Linear preserves constructor RNG consumption.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * d_model - 1
=======
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates are
        # zero, the remaining first-row pivot is fixed nonzero, and one trailing
        # value coefficient is zero. The original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 8
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * d_model - 1
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.shared_value_weight_trim = 1
=======
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.query_pivot = 0.02
        self.qkv.shared_value_weight_trim = 1
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
=======
        query_weight_size = (
            d_model * d_model - self.qkv.query_weight_trim - 1
        )
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = torch.cat(
            (
                query_weight.new_zeros(self.qkv.query_weight_trim),
                query_weight.new_full((1,), self.qkv.query_pivot),
                query_weight,
            )
        ).view(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
=======
                            query_weight.flatten()[
                                module.query_weight_trim + 1 :
                            ],
                            key_weight[: module.shared_kv_dim].flatten(),
>>>>>>> REPLACE