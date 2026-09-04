MECHANISM: Fourth LayerNorm-induced key-row gauge quotient

HYPOTHESIS: Extending the qualified 1,605-parameter design with a fourth compact key-weight row will yield 1,604 parameters and at least 99% accuracy, because centering the LayerNorm-scaled key row changes every key by only an attention-softmax-invariant constant.

INTENDED_EDIT: Reproduce the verified two-coordinate `ln1.bias` compaction, then Helmert-parameterize one additional key row while preserving the full reconstructed QKV projection.

EVIDENCE: The two-coordinate `ln1.bias` design achieved 99.54% at 1,605 parameters, whereas fixing a third coordinate fell to 96.07%; the successful designs already contain three instances of the independent key-row gauge, motivating extension of that exact attention invariance instead.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with three LayerNorm-induced key-weight gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four LayerNorm-induced key-weight gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                linear.weight[second_head_start + 1 :],
=======
                linear.weight[second_head_start + 2 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
            linear.weight[[key_start, key_start + 1, second_head_start]]
=======
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=1,
        )
=======
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix three key-row
        # gauges, and quotient one independently biased MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges, and quotient one independently biased MLP input row.
>>>>>>> REPLACE