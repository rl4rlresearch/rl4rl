MECHANISM: Direct full-rank affine bilinear attention scoring

HYPOTHESIS: Replacing each head’s separate query/key factors with its observable full-rank affine bilinear score operator will reduce parameters from 1,109 to 1,101 while retaining at least 99% accuracy, because it exactly preserves initialized attention scores, keeps head-specific routing, and enlarges rather than restricts the learned score-function class.

INTENDED_EDIT: Compose initialized query, key, and query-bias parameters into independent learned 8-by-7 score operators per head; compute causal attention directly in the normalized seven-dimensional residual subspace while leaving values and all other computation unchanged.

EVIDENCE: The 1,109-parameter model achieved 99.93%, whereas shared keys collapsed to 56.39% and an earlier restrictive query/key refactor reached 90.12%. This indicates that expressive head-specific score maps are load-bearing; the proposed composite operators preserve them without retaining redundant internal query/key coordinates.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias and independent query/key projections while
        # sharing one zero-mean value readout across the routing heads.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
=======
        # Construct the original affine first to preserve the baseline RNG
        # stream. After initialization, compose each head's query, key, and
        # query bias into the directly observable affine bilinear score map.
        # This retains independent head routing without redundant Q/K factors.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def gauge_fix_qkv(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.qkv.weight
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            # Combine the initialized per-head value maps at variance-preserving
            # scale, then learn a single semantic readout used by both routes.
            v_weight = v_weight.view(
                self.n_head, self.head_dim, d_model
            ).sum(dim=0) / math.sqrt(self.n_head)
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_coeff.reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
=======
    def gauge_fix_qkv(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.qkv.weight
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]

            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = (q_weight @ self.proj_basis).view(
                self.n_head, self.head_dim, d_model - 1
            )
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = (k_weight @ self.proj_basis).view(
                self.n_head, self.head_dim, d_model - 1
            )
            q_bias = self.qkv.bias.view(self.n_head, self.head_dim)

            # On zero-mean normalized inputs, QK^T and the query-bias term are
            # completely described by one affine bilinear operator per head.
            # Learning that full operator removes the factorization gauge and
            # also permits ranks above the original head dimension.
            score_bilinear = q_coeff.transpose(1, 2) @ k_coeff
            score_linear = (q_bias.unsqueeze(1) @ k_coeff).squeeze(1)
            score_coeff = torch.cat(
                (score_bilinear, score_linear.unsqueeze(1)), dim=1
            )

            # Combine the initialized per-head value maps at variance-preserving
            # scale, then learn a single semantic readout used by both routes.
            v_weight = v_weight.view(
                self.n_head, self.head_dim, d_model
            ).sum(dim=0) / math.sqrt(self.n_head)
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        score_coeff.reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
            self.qkv.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        matrix_size = d_model * (d_model - 1)
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_coeff = self.qkv.weight[matrix_size : 2 * matrix_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight, self.qkv.bias)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        score_size = self.n_head * d_model * (d_model - 1)
        score_coeff = self.qkv.weight[:score_size].view(
            self.n_head, d_model, d_model - 1
        )
        v_coeff = self.qkv.weight[score_size:].view(
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        v = F.linear(x, v_weight)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)

        content = x @ self.proj_basis
        affine_content = torch.cat(
            (content, content.new_ones(bsz, seqlen, 1)), dim=-1
        )
        routed_query = torch.einsum(
            "bti,hij->bhtj", affine_content, score_coeff
        )
        att = (
            routed_query @ content.transpose(1, 2).unsqueeze(1)
        ) / math.sqrt(self.head_dim)
>>>>>>> REPLACE