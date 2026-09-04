MECHANISM: Batched orthonormal attention-null reuse

HYPOTHESIS: Reusing projection column 3’s normalized row-common component as the remaining final-LayerNorm scale will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy; batching the four null-component reductions into one operation may avoid the prior attempt’s verification timeout.

INTENDED_EDIT: Remove the dedicated final-LayerNorm scale, initialize projection column 3’s null component to represent scale one, and derive all shared final-normalization values from one batched column-mean reduction.

EVIDENCE: Orthonormal null reuse achieved 100% accuracy for a final-LayerNorm scale at 1,267 parameters and its bias at 1,266. The identical third reuse previously timed out without producing contrary accuracy evidence.

<<<<<<< SEARCH
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
=======
        self.normalized_shape = (d_model,)
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
=======
        weight = torch.cat(
            (
                shared_scales[2:3],
                shared_scales.new_ones(1),
                shared_scales[1:2],
                shared_scales.new_ones(2),
                shared_scales[:1],
                shared_scales.new_ones(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
=======
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
            final_scale_column = self.blocks[-1].attn.proj.weight[:, 3]
            final_scale_column.add_(
                target_mean - final_scale_column.mean()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        final_proj_weight = self.blocks[-1].attn.proj.weight
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
        shared_final_bias = (
            math.sqrt(self.cfg.d_model) * final_proj_weight[:, 2].mean()
        )
=======
        final_proj_weight = self.blocks[-1].attn.proj.weight
        shared_nulls = final_proj_weight[:, :4].mean(dim=0)
        shared_final_scales = torch.stack(
            (
                shared_nulls[0],
                math.sqrt(self.cfg.d_model) * shared_nulls[1],
                math.sqrt(self.cfg.d_model) * shared_nulls[3],
            )
        )
        shared_final_bias = math.sqrt(self.cfg.d_model) * shared_nulls[2]
>>>>>>> REPLACE