MECHANISM: Attention residual-output matrix quotient

HYPOTHESIS: Removing the feature-uniform output coordinate from each of the eight attention projection columns will reduce the verified 1558-parameter model to 1550 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store `attn.proj.weight` as eight rows of seven relative output coordinates, reconstruct its full matrix during inference, and preserve full-coordinate AdamW, clipping, and value-bias absorption dynamics.

EVIDENCE: The current design achieved 99.96% after applying the identical per-column quotient to all 12 `fc2.weight` columns; attention projection columns have the same final-LayerNorm-canceled residual-output invariance.

<<<<<<< SEARCH
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        weight_relative = torch.cat(
            (
                self.proj.weight,
                self.proj.weight.new_zeros(
                    (self.proj.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        proj_weight = (
            weight_relative
            + self.proj.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Each MLP-output column is defined only modulo a feature-uniform
        # vector, which the final LayerNorm cancels. Store relative output
        # coordinates with the hidden dimension first.
        for block in self.blocks:
            full_fc2_weight = block.mlp.fc2.weight.detach().transpose(0, 1)
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )
=======
        # Each residual-output projection column is defined only modulo a
        # feature-uniform vector canceled by downstream LayerNorms. Store
        # relative output coordinates with the input dimension first.
        for block in self.blocks:
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1)
            )
            block.attn.proj.weight = nn.Parameter(
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            full_fc2_weight = block.mlp.fc2.weight.detach().transpose(0, 1)
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuotientAdamW(torch.optim.AdamW):
=======
def reconstruct_output_weight(param):
    relative = torch.cat(
        (
            param,
            param.new_zeros((param.size(0), 1)),
        ),
        dim=-1,
    )
    return (
        relative + param.mean(dim=-1, keepdim=True)
    ).transpose(0, 1)


class QuotientAdamW(torch.optim.AdamW):
>>>>>>> REPLACE

<<<<<<< SEARCH
                grad = (
                    proj_weight.detach()
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
=======
                full_proj_weight = reconstruct_output_weight(
                    proj_weight.detach()
                )
                grad = (
                    full_proj_weight
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_delta = proj_weight @ omitted_value
=======
            full_delta = (
                reconstruct_output_weight(proj_weight)
                @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        omitted_grad = (
            proj_weight.detach()
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
=======
        full_proj_weight = reconstruct_output_weight(
            proj_weight.detach()
        )
        omitted_grad = (
            full_proj_weight
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
=======
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.attn.proj.weight for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
>>>>>>> REPLACE