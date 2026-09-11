MECHANISM: Gauge-fixed shared-value chart

HYPOTHESIS: Fixing a full-rank 0.02-scaled identity block in the shared-value projection and restoring the unconstrained attention-output projection will retain at least 99% accuracy while reducing the model from 1,140 to 1,126 learned parameters.

INTENDED_EDIT: Parameterize the shared 4×8 value dictionary as a fixed 0.02I block plus a learned 4×4 block, restore the full output projection, and evaluate shared keys and values before head replication.

EVIDENCE: The 1,138-parameter factorized shared-KV design reached 99.78%, while zeroing an entire value block failed because it removed the value route. A fixed full-rank block instead uses the reciprocal value/output basis freedom without reducing the value dictionary’s rank, and restoring the sensitive output coefficients avoids the failures caused by further output pruning.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates and
        # one trailing value coefficient are gauge-fixed. Constructing the
        # original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * d_model - 1
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_weight_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.shared_value_weight_trim = 1
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Fix one coefficient on the output side of the shared-value basis.
        # Constructing the full Linear preserves constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 1))
        self.proj.fixed_weight_trim = (0, 1)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates are
        # gauge-fixed. The shared value dictionary uses a dense gauge chart:
        # its leading square block is a fixed nonzero identity and its remaining
        # block is learned. The original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * (d_model - self.head_dim)
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_weight_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.shared_value_free_start = self.head_dim
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.register_buffer(
            "value_anchor",
            0.02 * torch.eye(self.head_dim),
            persistent=False,
        )
        self.proj = nn.Linear(d_model, d_model)
        # The full output map absorbs changes of basis in the canonicalized
        # shared-value dictionary.
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model).repeat(self.n_head, 1)
        qkv_weight = torch.cat((query_weight, key_weight, value_weight), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        value_weight_size = self.head_dim * (d_model - self.head_dim)
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                value_weight_size,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model)
        value_weight = value_weight.view(
            self.head_dim,
            d_model - self.head_dim,
        )
        value_weight = torch.cat((self.value_anchor, value_weight), dim=1)

        q = F.linear(x, query_weight, query_bias)
        shared_kv = F.linear(
            x,
            torch.cat((key_weight, value_weight), dim=0),
        )
        k, v = shared_kv.chunk(2, dim=-1)
        k = k.repeat(1, 1, self.n_head)
        v = v.repeat(1, 1, self.n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = F.pad(self.proj.weight, (0, 1)).view(d_model, d_model)
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, proj_weight, proj_bias)
=======
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, self.proj.weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
                            value_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_value_weight_trim],
=======
                            value_weight[
                                : module.shared_kv_dim,
                                module.shared_value_free_start :,
                            ].flatten(),
>>>>>>> REPLACE