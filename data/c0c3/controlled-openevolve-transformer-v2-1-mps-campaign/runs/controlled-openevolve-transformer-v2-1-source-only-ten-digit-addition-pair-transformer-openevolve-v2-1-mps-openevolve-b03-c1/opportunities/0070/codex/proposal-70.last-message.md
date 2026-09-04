MECHANISM: Multi-query attention with shared learned keys and values

HYPOTHESIS: Sharing the four-dimensional key representation across both heads will reduce the model from 1,118 to 1,090 parameters while retaining at least 99% accuracy, because head specialization can remain in the independent queries, output blocks, and load-bearing per-head relative biases.

INTENDED_EDIT: Replace the two independent 28-parameter zero-mean key maps with one variance-preserving shared key map, producing multi-query attention while preserving independent queries and all routing biases.

EVIDENCE: The current design already succeeds with a shared value representation, while asymmetric endpoint pruning shows that head-specific distance routing is load-bearing. This challenges the shared assumption that semantic key features must also be head-specific without removing the demonstrated source of head specialization.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias and independent query/key projections while
        # sharing one zero-mean value readout across the routing heads.
=======
        # Construct the original affine first to preserve the baseline RNG
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias and independent query projections while
        # sharing zero-mean key and value readouts across the routing heads.
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            # A common semantic key dictionary leaves route specialization in
            # the independent queries and relative biases. Combine initialized
            # heads at variance-preserving scale.
            k_weight = k_weight.view(
                self.n_head, self.head_dim, d_model
            ).sum(dim=0) / math.sqrt(self.n_head)
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            # Use the same multi-query sharing pattern for values.
            v_weight = v_weight.view(
                self.n_head, self.head_dim, d_model
            ).sum(dim=0) / math.sqrt(self.n_head)
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        matrix_size = d_model * (d_model - 1)
        shared_matrix_size = self.head_dim * (d_model - 1)
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_coeff = self.qkv.weight[
            matrix_size : matrix_size + shared_matrix_size
        ].view(self.head_dim, d_model - 1)
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[
            matrix_size + shared_matrix_size :
        ].view(self.head_dim, d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)
=======
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)
>>>>>>> REPLACE