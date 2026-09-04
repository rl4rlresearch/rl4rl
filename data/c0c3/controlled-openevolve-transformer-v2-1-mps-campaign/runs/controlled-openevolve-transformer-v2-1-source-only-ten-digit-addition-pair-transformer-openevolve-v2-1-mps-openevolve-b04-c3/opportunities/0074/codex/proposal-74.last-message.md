MECHANISM: Three-entry positional-mixer constraint

HYPOTHESIS: Fixing a third scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,217 to 1,216 learned parameters.

INTENDED_EDIT: Adopt the qualified two-entry positional-mixer constraint, fix one additional final mixer weight at zero, and preserve full-matrix initialization RNG consumption.

EVIDENCE: Reference Design 2 achieved 100% accuracy with two fixed positional-mixer scalars and 1,217 parameters; since the aggressive eight-gain restriction failed at 46.49%, the most informative next change is one additional scalar constraint while retaining dense cross-channel mixing.

<<<<<<< SEARCH
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
=======
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.fixed_weight_trim = (7, 0)
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Positions are represented by a generic fixed Fourier coordinate
        # system and synthesized by one learned map shared across all steps.
        # This replaces an independent d_model-vector lookup at every position.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
=======
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final three scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 3)
        )
        self.pos_mix.fixed_weight_trim = (0, 3)
>>>>>>> REPLACE

<<<<<<< SEARCH
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[7:])
=======
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_x = self.pos_mix(pos_features.to(token_x.dtype))
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 3)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
        pos_x = F.linear(pos_features.to(token_x.dtype), pos_weight)
>>>>>>> REPLACE