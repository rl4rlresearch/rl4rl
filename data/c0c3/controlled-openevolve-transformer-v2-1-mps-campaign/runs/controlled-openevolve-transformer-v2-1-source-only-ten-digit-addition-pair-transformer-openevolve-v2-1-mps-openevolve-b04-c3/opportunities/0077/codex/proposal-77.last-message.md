MECHANISM: Six-entry positional-mixer constraint

HYPOTHESIS: Fixing a sixth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,214 to 1,213 learned parameters.

INTENDED_EDIT: Learn 58 of the 64 positional-mixer weights, fixing the final six entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.

EVIDENCE: The five-entry positional-mixer design achieved 100% accuracy with 1,214 parameters, following 100% results for one through four fixed entries; this motivates the smallest incremental sixth constraint while retaining dense cross-channel mixing.

<<<<<<< SEARCH
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final five scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 5)
        )
        self.pos_mix.fixed_weight_trim = (0, 5)
=======
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final six scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 6)
        )
        self.pos_mix.fixed_weight_trim = (0, 6)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 5)).view(
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 6)).view(
>>>>>>> REPLACE