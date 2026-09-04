MECHANISM: Orthogonally mixed diagonal Fourier positional synthesis

HYPOTHESIS: Replacing the qualified dense 8×8 Fourier synthesizer with a fixed generic orthogonal transform and eight learned channel gains will retain at least 99% accuracy after 45,000 steps while reducing the qualified model from 1,219 to 1,163 learned parameters.

INTENDED_EDIT: Adopt the proven seven-entry query-weight gauge and Fourier positional backbone, then constrain positional synthesis to a fixed DCT basis with eight trainable gains while preserving initialization RNG consumption and the rank-six tied token code.

EVIDENCE: Reference Design 3 achieved 100% accuracy with 1,219 parameters using generic sinusoidal coordinates and a learned 8×8 projection; this tests whether downstream learned attention and MLP projections can supply the required coordinate mixing while positional learning retains only per-channel amplitudes. The failed five-coordinate token lift motivates leaving the qualified token representation unchanged.

<<<<<<< SEARCH
        # Key and value biases are redundant. Q/K basis changes permit four
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 4))
=======
        # Key and value biases are redundant. Q/K basis changes permit seven
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (4, 0)).view(3 * d_model, d_model)
=======
        qkv_weight = F.pad(self.qkv.weight, (7, 0)).view(3 * d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class DiagonalPositionMix(nn.Linear):
    """Fixed orthogonal feature mixing followed by learned channel gains."""

    def __init__(self, d_model: int):
        # Constructing the dense layer preserves the qualified constructor's
        # RNG consumption before replacing it with a diagonal parameterization.
        super().__init__(d_model, d_model, bias=False)
        self.weight = nn.Parameter(torch.empty(d_model))

        out_index = torch.arange(d_model, dtype=torch.float32).unsqueeze(1)
        in_index = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        basis = torch.cos(
            math.pi * (in_index + 0.5) * out_index / d_model
        )
        basis[0].mul_(1.0 / math.sqrt(2.0))
        basis.mul_(math.sqrt(2.0 / d_model))
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = F.linear(features, self.basis)
        return math.sqrt(self.in_features) * self.weight * mixed


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Generic sinusoidal coordinates are spread across residual channels by
        # a fixed orthogonal basis; only one gain per channel is learned.
        self.pos_mix = DiagonalPositionMix(cfg.d_model)
        pos_inv_freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        self.register_buffer("pos_inv_freq", pos_inv_freq, persistent=False)

        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, nn.Linear):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[4:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        if isinstance(module, DiagonalPositionMix):
            full_weight = module.weight.new_empty(
                module.out_features * module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    full_weight.view(
                        module.out_features,
                        module.in_features,
                    ).diagonal()
                )
        elif isinstance(module, nn.Linear):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[7:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        x = self.token_scale * token_x + self.pos_emb(pos)
        x = self.drop(x)
=======
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        angles = pos.to(self.pos_inv_freq.dtype).unsqueeze(-1) * self.pos_inv_freq
        pos_features = torch.stack(
            (angles.sin(), angles.cos()),
            dim=-1,
        ).flatten(-2)
        pos_x = self.pos_mix(pos_features.to(token_x.dtype))
        x = self.token_scale * token_x + pos_x
        x = self.drop(x)
>>>>>>> REPLACE