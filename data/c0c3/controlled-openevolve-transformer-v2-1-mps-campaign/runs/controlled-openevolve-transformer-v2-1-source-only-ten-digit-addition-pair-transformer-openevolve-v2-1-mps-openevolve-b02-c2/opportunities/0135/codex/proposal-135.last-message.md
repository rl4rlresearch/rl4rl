MECHANISM: Complete pre-MLP LayerNorm-bias absorption with lower-overhead training

HYPOTHESIS: Removing all `ln2` bias parameters will produce a 1,576-parameter model with at least 99% accuracy, while collision-loop and validation-overhead reductions will allow the previously timed-out design to complete.

INTENDED_EDIT: Represent fully fixed LayerNorm bias without a zero-length parameter, fix all eight `ln2` bias coordinates, accelerate exact holdout rejection, and validate only the final trained checkpoint.

EVIDENCE: The 1,577-parameter design reached 99.81% accuracy after fixing seven `ln2` bias coordinates; both full-absorption attempts timed out rather than failing accuracy, motivating the same exact gauge reduction with stronger runtime savings.

<<<<<<< SEARCH
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )
=======
        retained_bias = (
            layer_norm.bias[:-fixed_coordinates]
            if fixed_coordinates
            else layer_norm.bias
        )
        if retained_bias.numel():
            self.bias = nn.Parameter(retained_bias.detach().clone())
        else:
            self.register_parameter("bias", None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.fixed_coordinates))
        )
=======
        if self.bias is None:
            full_bias = full_weight.new_zeros(self.normalized_shape)
        else:
            full_bias = torch.cat(
                (self.bias, self.bias.new_zeros(self.fixed_coordinates))
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=2,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=cfg.d_model,
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Strictly avoid holdout leakage.
        for i in range(self.batch_size):
            ai = int(a[i].item())
            bi = int(b[i].item())
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
=======
        # Strictly avoid holdout leakage while keeping the overwhelmingly
        # collision-free path out of per-element tensor indexing.
        a_values = a.tolist()
        b_values = b.tolist()
        for i, (ai, bi) in enumerate(zip(a_values, b_values)):
            if pair_hash(ai, bi) not in self.reserved_hashes:
                continue
            while pair_hash(ai, bi) in self.reserved_hashes:
                ai = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
                bi = int(torch.randint(0, MAX_OPERAND, (1,), generator=self.g, dtype=torch.int64).item())
            a[i] = ai
            b[i] = bi
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE