MECHANISM: Shared-value, head-specific routing attention

HYPOTHESIS: Sharing one four-dimensional learned value projection across both heads will reduce parameters from 1,192 to 1,164 while retaining at least 99% accuracy, because addition requires distinct head-specific addressing but both routes can transport digits through a common learned value representation.

INTENDED_EDIT: Preserve independent query and key projections and biases for each head, but replace the two value projections with one learned shared projection whose attended outputs remain separately mixed by the output projection.

EVIDENCE: Shared-key attention collapsed to 36.45%, identifying head-specific routing as load-bearing, while the current two-head model reaches 99.99%; this patch leaves that routing capacity intact and tests the distinct assumption that each route also needs its own semantic value coordinates.

<<<<<<< SEARCH
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias while storing all three input projections in
        # the observable zero-mean LayerNorm subspace.
=======
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias and independent query/key projections while
        # sharing one zero-mean value readout across the routing heads.
>>>>>>> REPLACE

<<<<<<< SEARCH
            v_weight = weight[2 * d_model :]
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            d_model, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight, self.qkv.bias)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
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
>>>>>>> REPLACE