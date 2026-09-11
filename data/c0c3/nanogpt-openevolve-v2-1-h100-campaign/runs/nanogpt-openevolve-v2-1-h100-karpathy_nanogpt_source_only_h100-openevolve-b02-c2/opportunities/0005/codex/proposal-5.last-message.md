MECHANISM: Three-quarter-device-batch update cadence

HYPOTHESIS: A 196,608-token single-microbatch step will lower val_bpb below 0.987182 by providing 33% more updates than the successful 262K-token design while avoiding the severe throughput loss observed with a 64-sequence device batch.

INTENDED_EDIT: Restore the proven separate Q/K/V projections and optimizer path, then train with one 96×2048-token device batch per optimizer step.

EVIDENCE: The 262K-token single-microbatch design achieved the best val_bpb, 0.987182, at 489.7M tokens; reducing the device batch to 64 raised update frequency but collapsed throughput to 357.2M tokens and worsened val_bpb to 1.004099, motivating an intermediate 96-sequence batch.

<<<<<<< SEARCH
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        self.qkv_split_sizes = (q_dim, kv_dim, kv_dim)
        self.c_qkv = nn.Linear(self.n_embd, q_dim + 2 * kv_dim, bias=False)
=======
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = self.c_qkv(x).split(self.qkv_split_sizes, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
=======
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
            torch.nn.init.uniform_(block.attn.c_qkv.weight, -s, s)
=======
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
>>>>>>> REPLACE

<<<<<<< SEARCH
            chunks = 3 if shape == (3 * model_dim, model_dim) else 1
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                chunks=chunks,
            ))
=======
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
>>>>>>> REPLACE

<<<<<<< SEARCH
        updated_params = stacked_params.view(num_params, *raw_shape)
        torch._foreach_copy_(params, list(updated_params.unbind(0)))
=======
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step; one device batch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE