MECHANISM: Third independent query-coordinate gauge normalization

HYPOTHESIS: Fixing a pivot in the third query coordinate at 0.02 will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,137-parameter model to 1,136 learned parameters, because reciprocal scaling of the corresponding shared-key coordinate preserves attention scores.

INTENDED_EDIT: Preserve the two qualified query pivots and additionally reconstruct query-matrix index 16 as a fixed nonzero pivot, omitting its initialized value from the compact learned parameter.

EVIDENCE: The current design achieved 100% accuracy after independently fixing pivots for the first two query coordinates; applying the same gauge normalization to the next query coordinate is the smallest evidence-backed reduction and avoids constraining another coefficient of an already-normalized coordinate.

<<<<<<< SEARCH
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coefficients are
        # zero, the next two coordinate pivots are fixed nonzero, and one
        # trailing value coefficient is zero. Constructing the original Linear
        # preserves constructor RNG.
        query_weight_size = d_model * d_model - 9
=======
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coefficients are
        # zero, pivots for the first three query coordinates are fixed nonzero,
        # and one trailing value coefficient is zero. Constructing the original
        # Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 10
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv.query_weight_trim = 7
        self.qkv.query_fixed_pivots = 2
        self.qkv.query_pivot = 0.02
=======
        self.qkv.query_weight_trim = 7
        self.qkv.query_fixed_pivots = 2
        self.qkv.query_third_pivot_offset = 7
        self.qkv.query_pivot = 0.02
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_weight_size = (
            d_model * d_model
            - self.qkv.query_weight_trim
            - self.qkv.query_fixed_pivots
        )
=======
        query_weight_size = (
            d_model * d_model
            - self.qkv.query_weight_trim
            - self.qkv.query_fixed_pivots
            - 1
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_weight = torch.cat(
            (
                query_weight.new_zeros(self.qkv.query_weight_trim),
                query_weight.new_full(
                    (self.qkv.query_fixed_pivots,),
                    self.qkv.query_pivot,
                ),
                query_weight,
            )
        ).view(d_model, d_model)
=======
        query_weight = torch.cat(
            (
                query_weight.new_zeros(self.qkv.query_weight_trim),
                query_weight.new_full(
                    (self.qkv.query_fixed_pivots,),
                    self.qkv.query_pivot,
                ),
                query_weight[: self.qkv.query_third_pivot_offset],
                query_weight.new_full((1,), self.qkv.query_pivot),
                query_weight[self.qkv.query_third_pivot_offset :],
            )
        ).view(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    compact_weight = torch.cat(
                        (
                            query_weight.flatten()[
                                module.query_weight_trim
                                + module.query_fixed_pivots :
                            ],
                            key_weight[: module.shared_kv_dim].flatten(),
=======
                    query_start = (
                        module.query_weight_trim
                        + module.query_fixed_pivots
                    )
                    third_pivot = (
                        query_start + module.query_third_pivot_offset
                    )
                    compact_weight = torch.cat(
                        (
                            query_weight.flatten()[query_start:third_pivot],
                            query_weight.flatten()[third_pivot + 1 :],
                            key_weight[: module.shared_kv_dim].flatten(),
>>>>>>> REPLACE