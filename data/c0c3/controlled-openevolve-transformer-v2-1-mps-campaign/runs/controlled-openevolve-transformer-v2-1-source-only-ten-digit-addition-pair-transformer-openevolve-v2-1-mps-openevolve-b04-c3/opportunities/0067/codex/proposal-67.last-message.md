MECHANISM: Five-entry query/key basis gauge fixing

HYPOTHESIS: Fixing a fifth leading query-weight coordinate at zero on the qualified 1,342-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,341 learned parameters.

INTENDED_EDIT: Extend the proven query-weight constraint from four to five leading flattened entries, reconstructing the full QKV matrix with five fixed zeros while preserving initialization RNG consumption.

EVIDENCE: The four-entry design achieved 100% accuracy with 1,342 parameters, following 100% results with one, two, and three fixed entries; this directly motivates testing the next scalar in the same Q/K basis gauge.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes permit the
        # query bias to be shared across heads with one coordinate fixed at
        # zero. Constructing the original Linear first preserves its RNG use.
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. Q/K basis changes permit five
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 5))
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        qkv_weight = F.pad(self.qkv.weight, (5, 0)).view(3 * d_model, d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
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
                    module.weight.copy_(full_weight[5:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE