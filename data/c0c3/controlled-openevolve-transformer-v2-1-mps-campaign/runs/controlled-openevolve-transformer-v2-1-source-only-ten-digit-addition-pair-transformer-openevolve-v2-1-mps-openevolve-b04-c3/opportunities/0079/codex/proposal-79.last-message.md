MECHANISM: Eight-entry positional-mixer constraint

HYPOTHESIS: Fixing an eighth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,212 to 1,211 learned parameters.

INTENDED_EDIT: Learn 56 of the 64 positional-mixer weights, fixing the final eight entries at zero while preserving full-matrix initialization RNG consumption and the qualified transformer backbone.

EVIDENCE: Reference Design 3 achieved 100% accuracy with seven fixed positional-mixer scalars and 1,212 parameters, following successful one-through-six-scalar constraints; this motivates the smallest incremental eighth constraint while retaining learned cross-channel positional mixing.

<<<<<<< SEARCH
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final three scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 3)
        )
        self.pos_mix.fixed_weight_trim = (0, 3)
=======
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final eight scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 8)
        )
        self.pos_mix.fixed_weight_trim = (0, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 3)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 8)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
>>>>>>> REPLACE