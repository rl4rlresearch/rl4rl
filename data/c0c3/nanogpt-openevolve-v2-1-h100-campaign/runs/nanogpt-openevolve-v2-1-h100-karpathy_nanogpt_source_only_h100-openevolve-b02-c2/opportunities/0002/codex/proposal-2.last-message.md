MECHANISM: Optimizer-preserving fused QKV projection

HYPOTHESIS: Fusing each layer’s three attention input projections into one GEMM will preserve model capacity and Muon’s per-matrix updates while increasing throughput beyond 497.0M tokens, lowering val_bpb below 0.995558.

INTENDED_EDIT: Replace separate Q, K, and V linear layers with one fused projection, then reshape fused gradients into three independent matrices during Muon orthogonalization.

EVIDENCE: Quarter-context attention reduced throughput from 497.0M to 385.4M tokens and worsened val_bpb to 1.012584, motivating an execution-level optimization that retains the successful 1024/2048-token attention pattern.

<<<<<<< SEARCH
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
=======
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        self.qkv_split_sizes = (q_dim, kv_dim, kv_dim)
        self.c_qkv = nn.Linear(self.n_embd, q_dim + 2 * kv_dim, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
=======
        q, k, v = self.c_qkv(x).split(self.qkv_split_sizes, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
=======
            torch.nn.init.uniform_(block.attn.c_qkv.weight, -s, s)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            chunks = 3 if shape == (3 * model_dim, model_dim) else 1
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                chunks=chunks,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
=======
    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        raw_shape, device, dtype = p.shape, p.device, p.dtype
        chunks = group.get("chunks", 1)
        assert raw_shape[-2] % chunks == 0
        shape = (raw_shape[-2] // chunks, raw_shape[-1])
        num_matrices = num_params * chunks
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_matrices, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_matrices, shape[-2], 1) if shape[-2] >= shape[-1] else (num_matrices, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params]).view(num_matrices, *shape)
        stacked_params = torch.stack(params).view(num_matrices, *shape)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        updated_params = stacked_params.view(num_params, *raw_shape)
        torch._foreach_copy_(params, list(updated_params.unbind(0)))
>>>>>>> REPLACE