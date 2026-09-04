MECHANISM: Function-preserving value/output coordinate-scale gauge

HYPOTHESIS: Fixing `qkv.weight[16,0]` at 0.02 while reciprocally scaling output-projection column 0 will reduce the model to 1576 parameters and retain at least 99% accuracy because it preserves the initialized network function and uses the independent value/output gauge instead of the optimization-sensitive fourth head-0 query gauge.

INTENDED_EDIT: Store 182 learned QKV weights, reconstruct value weight 128 as 0.02, scale value row 16 during initialization, and inversely scale output-projection column 0 after its ordinary initialization.

EVIDENCE: Seven function-preserving query/key scale anchors produced the verified 1577-parameter model at 99.13%, while both tested versions of a fourth head-0 query anchor failed; moving to the analogous independent value/output scale symmetry is the most informative next reduction.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 9))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with query/key anchors and one value scale anchor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[41:48],
                fixed,
                self.weight_rest[48:],
=======
                fixed,
                self.weight_rest[41:48],
                fixed,
                self.weight_rest[48:119],
                fixed,
                self.weight_rest[119:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                flat = weight.flatten()
=======
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                value_scale = 0.02 / weight[16, 0]
                weight[16].mul_(value_scale)
                module._value_init_scale = value_scale.item()

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[49:56],
                            flat[57:],
=======
                            flat[49:56],
                            flat[57:128],
                            flat[129:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                    )
                )
        elif isinstance(module, SharedAnchorEmbeddings):
=======
                    )
                )
        elif isinstance(module, CausalSelfAttention):
            # Complete the value/output gauge transform after the ordinary
            # output projection has received its random initialization.
            with torch.no_grad():
                module.proj.weight[:, 0].div_(module.qkv._value_init_scale)
        elif isinstance(module, SharedAnchorEmbeddings):
>>>>>>> REPLACE