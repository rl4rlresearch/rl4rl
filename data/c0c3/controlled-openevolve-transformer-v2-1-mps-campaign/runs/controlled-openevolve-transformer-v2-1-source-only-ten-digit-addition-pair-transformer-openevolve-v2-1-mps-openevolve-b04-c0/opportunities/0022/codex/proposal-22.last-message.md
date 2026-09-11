MECHANISM: Distributed sixth one-sided embedding gauge anchor

HYPOTHESIS: Adding a detached-position sixth anchor between `token[3, 5]` and `position[0, 5]` will reduce the model from 1623 to 1622 parameters while retaining at least 99% accuracy, because the independently distributed fourth and fifth anchors achieved 99.98% and 99.94%, respectively, without concentrating another constraint on an existing token row.

INTENDED_EDIT: Expand the shared embedding representation to six anchors, place the new coordinate on token row 3, detach its positional gradient, and preserve the gauge-equivalent initialization.

EVIDENCE: Distributing the fourth detached anchor to token row 1 recovered accuracy from 74.22% to 99.98%, and distributing the fifth to token row 2 retained 99.94%; this directly supports testing the next unused token row and coordinate.

<<<<<<< SEARCH
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing five gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(5))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 5))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 5))
=======
class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing six gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(6))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 6))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
        fifth_index = 2 * self.d_model + 4
        fourth_offset = fourth_index - 3
        fifth_offset = fourth_offset + fifth_index - fourth_index - 1
        flat = torch.cat(
            (
                self.anchor[:3],
                self.token_rest[:fourth_offset],
                self.anchor[3:4],
                self.token_rest[fourth_offset:fifth_offset],
                self.anchor[4:],
                self.token_rest[fifth_offset:],
            )
        )
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third through fifth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
=======
    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
        fifth_index = 2 * self.d_model + 4
        sixth_index = 3 * self.d_model + 5
        fourth_offset = fourth_index - 3
        fifth_offset = fourth_offset + fifth_index - fourth_index - 1
        sixth_offset = fifth_offset + sixth_index - fifth_index - 1
        flat = torch.cat(
            (
                self.anchor[:3],
                self.token_rest[:fourth_offset],
                self.anchor[3:4],
                self.token_rest[fourth_offset:fifth_offset],
                self.anchor[4:5],
                self.token_rest[fifth_offset:sixth_offset],
                self.anchor[5:],
                self.token_rest[sixth_offset:],
            )
        )
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third through sixth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
            fifth_shift = 0.5 * (pos[0, 4] - token[2, 4])
            token[:, 4].add_(fifth_shift)
            pos[:, 4].sub_(fifth_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            token_flat = token.flatten()
            with torch.no_grad():
                module.anchor.copy_(
                    torch.cat((token[0, :3], token[1, 3:4], token[2, 4:5]))
                )
                module.token_rest.copy_(
                    torch.cat(
                        (
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[5:])
=======
            fifth_shift = 0.5 * (pos[0, 4] - token[2, 4])
            token[:, 4].add_(fifth_shift)
            pos[:, 4].sub_(fifth_shift)

            sixth_shift = 0.5 * (pos[0, 5] - token[3, 5])
            token[:, 5].add_(sixth_shift)
            pos[:, 5].sub_(sixth_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            sixth_index = 3 * module.d_model + 5
            token_flat = token.flatten()
            with torch.no_grad():
                module.anchor.copy_(
                    torch.cat(
                        (
                            token[0, :3],
                            token[1, 3:4],
                            token[2, 4:5],
                            token[3, 5:6],
                        )
                    )
                )
                module.token_rest.copy_(
                    torch.cat(
                        (
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 : sixth_index],
                            token_flat[sixth_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[6:])
>>>>>>> REPLACE