MECHANISM: Query–key inverse-scale gauge anchoring

HYPOTHESIS: Fixing one query-projection coefficient at the initialization scale will reduce the model from 1,571 to 1,570 learned parameters while retaining at least 99% accuracy after 21,000 training steps, because inverse rescaling of the corresponding query and key channels leaves attention scores unchanged.

INTENDED_EDIT: Split the first query-projection row from the dense QKV matrix, represent seven of its coefficients as learned parameters, and fix its eighth coefficient at 0.02 while preserving standard initialization for every learned coefficient.

EVIDENCE: The current 1,571-parameter model achieved 99.76% accuracy, whereas extending positional-row and attention-output shift gauges failed at 84.96%, 70%, and 40.36%; this tests a different exact attention symmetry without further extending those sensitive parameterizations.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 16))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv._defer_query_scale_gauge = True
        self.qkv.weight = nn.Parameter(
            torch.empty(3 * d_model - 1, d_model)
        )
        self.qkv.query_row_0 = nn.Parameter(torch.empty(d_model - 1))
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 16))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        query_row_0 = torch.cat(
            (
                self.qkv.query_row_0,
                self.qkv.query_row_0.new_full((1,), 0.02),
            )
        ).unsqueeze(0)
        qkv_weight = torch.cat((query_row_0, self.qkv.weight), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_defer_two_column_gauge", False):
                initialized_weight = module.weight.new_empty(
                    module.out_features, module.in_features - 1
                )
                nn.init.normal_(initialized_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight[:, :-1].copy_(initialized_weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_defer_query_scale_gauge", False):
                initialized_weight = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(initialized_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.query_row_0.copy_(initialized_weight[0, :-1])
                    module.weight.copy_(initialized_weight[1:])
            elif getattr(module, "_defer_two_column_gauge", False):
                initialized_weight = module.weight.new_empty(
                    module.out_features, module.in_features - 1
                )
                nn.init.normal_(initialized_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight[:, :-1].copy_(initialized_weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE