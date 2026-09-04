MECHANISM: Seven-entry positional-mixer constraint

HYPOTHESIS: Fixing a seventh scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,213 to 1,212 learned parameters.

INTENDED_EDIT: Learn 57 of the 64 positional-mixer weights, fixing the final seven entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.

EVIDENCE: Reference Design 1 achieved 100% accuracy with six fixed positional-mixer scalars and 1,213 parameters, following successful one-through-five-scalar constraints; this motivates the smallest incremental seventh constraint while retaining dense cross-channel mixing.

<<<<<<< SEARCH
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final four scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 4)
        )
        self.pos_mix.fixed_weight_trim = (0, 4)
=======
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final seven scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 7)
        )
        self.pos_mix.fixed_weight_trim = (0, 7)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 4)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 7)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
>>>>>>> REPLACE