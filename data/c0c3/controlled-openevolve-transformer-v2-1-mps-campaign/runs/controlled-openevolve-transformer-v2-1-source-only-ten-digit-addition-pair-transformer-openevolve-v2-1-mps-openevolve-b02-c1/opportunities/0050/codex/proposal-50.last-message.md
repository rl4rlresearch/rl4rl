MECHANISM: Independent learned affine positional pointers

HYPOTHESIS: Replacing each head’s unconstrained relative-distance table with its own learned affine pointer and learned sharpness will reduce parameters by `2*(INPUT_LEN-1)-6` while retaining at least 99% accuracy; head independence, rather than a free parameter at every distance, is the load-bearing positional requirement.

INTENDED_EDIT: Each attention head learns a Gaussian-shaped positional focus whose center moves affinely with the query position; remove the relative-bias tables and their quotient-optimizer handling while preserving all token, value, nonlinear, and decoding pathways.

EVIDENCE: Sharing one signed positional table collapsed accuracy to 30.51%, showing that distinct head roles are essential. The width-1 model nevertheless reached 100%, suggesting the successful computation is highly structured; independent learned affine pointers preserve separate operand-alignment roles while testing whether arbitrary per-distance tables are unnecessary.

<<<<<<< SEARCH
        # One bias per head is a softmax-invisible common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
=======
        # Each head learns an independent affine positional pointer.  The
        # endpoint-centered offset keeps the learned coordinates small while
        # imposing no operand-specific alignment.
        self.pointer_offset = nn.Parameter(torch.zeros(n_head))
        self.pointer_slope = nn.Parameter(torch.zeros(n_head))
        self.pointer_log_sharpness = nn.Parameter(
            torch.full((n_head,), -6.0)
        )
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        rel_bias = F.pad(self.rel_bias, (0, 1))
        att = att + rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
=======
        positions = torch.arange(
            seqlen, device=x.device, dtype=att.dtype
        )
        pointer_center = (
            float(self.mask.size(0) - 1)
            + self.pointer_offset[:, None]
            + self.pointer_slope[:, None] * positions[None, :]
        )
        pointer_distance = (
            positions[None, None, :] - pointer_center[:, :, None]
        )
        pointer_sharpness = F.softplus(
            self.pointer_log_sharpness
        )[:, None, None]
        att = att - pointer_sharpness * pointer_distance.square()
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
=======
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ]
>>>>>>> REPLACE