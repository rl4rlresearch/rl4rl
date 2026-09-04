MECHANISM: Bias-localized key–value sharing

HYPOTHESIS: Tying the bias-free second head’s key and value projections will reduce the model from 1609 to 1577 parameters while retaining at least 99% accuracy, because the first head retains fully independent projections and all three load-bearing value-bias coordinates.

INTENDED_EDIT: Challenge the assumption that every attention head needs separate key and value maps by giving the second head a shared learned associative representation, while preserving the fully general bias-carrying first head.

EVIDENCE: The current model achieved 100% after removing every query- and value-bias coordinate from the second head, whereas reducing the first head’s value bias from three coordinates to two collapsed accuracy to 55.14%; this identifies the first head as the safer place to preserve capacity and the bias-free second head as the informative target for sharing.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        # The first head keeps independent Q/K/V maps. In the bias-free
        # second head, the learned value features also serve as keys.
        self.qv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.k_primary = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 6))
        v = v + F.pad(self.v_bias, (0, 5))
=======
        qv = self.qv(x)
        q, v = qv.chunk(2, dim=-1)
        k = torch.cat((self.k_primary(x), v[..., self.head_dim :]), dim=-1)
        q = q + F.pad(self.q_bias, (0, 6))
        v = v + F.pad(self.v_bias, (0, 5))
>>>>>>> REPLACE