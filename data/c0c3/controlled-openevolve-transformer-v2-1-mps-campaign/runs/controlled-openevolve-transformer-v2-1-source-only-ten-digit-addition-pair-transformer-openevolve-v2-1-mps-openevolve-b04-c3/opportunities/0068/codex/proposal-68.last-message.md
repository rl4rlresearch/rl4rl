MECHANISM: Six-entry query/key basis gauge fixing

HYPOTHESIS: Fixing a sixth leading query-weight coordinate at zero on the qualified 1,341-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,340 learned parameters.

INTENDED_EDIT: Reproduce the qualified five-entry design, then extend its query-weight constraint to six leading flattened entries while preserving full-matrix initialization RNG consumption.

EVIDENCE: The five-entry design achieved 100% accuracy with 1,341 parameters, following 100% results with one through four fixed entries; this directly motivates testing the next scalar in the same Q/K basis gauge.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes allow the
        # query bias to be shared across heads and one shared coordinate to be
        # fixed at zero. Construct the original Linear first to preserve its
        # constructor RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. Q/K basis changes permit six
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 6))
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # The two token-free residual channels provide one rotational gauge.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        qkv_weight = F.pad(self.qkv.weight, (6, 0)).view(3 * d_model, d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
=======
        proj_bias = F.pad(self.proj.bias, (0, 1))
        y = F.linear(y, self.proj.weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[6:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE