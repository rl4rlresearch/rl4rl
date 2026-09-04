MECHANISM: Two-entry positional-mixer constraint

HYPOTHESIS: Fixing a second scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,218 to 1,217 learned parameters.

INTENDED_EDIT: Learn 62 of the 64 positional-mixer weights, fixing its final two entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.

EVIDENCE: The immediately preceding single-entry positional-mixer constraint achieved 100% accuracy with 1,218 parameters, while the eight-gain restriction failed at 46.49%; the smallest informative next test is one additional fixed mixer scalar.

<<<<<<< SEARCH
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final scalar is fixed at zero while all other channel couplings
        # remain learned. Constructing the full Linear preserves RNG use.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 1)
        )
        self.pos_mix.fixed_weight_trim = (0, 1)
=======
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final two scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 2)
        )
        self.pos_mix.fixed_weight_trim = (0, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 1)).view(
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 2)).view(
>>>>>>> REPLACE