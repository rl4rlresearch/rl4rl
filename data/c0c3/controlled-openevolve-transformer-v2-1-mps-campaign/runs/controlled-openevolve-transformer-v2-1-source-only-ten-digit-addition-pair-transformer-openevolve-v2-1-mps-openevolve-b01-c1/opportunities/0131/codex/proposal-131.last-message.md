MECHANISM: Single-channel hybrid positional encoding

HYPOTHESIS: Replacing only the sixth learned positional lookup profile with a scaled mean-free cosine will reduce parameters from 1,524 to 1,503 while retaining at least 99% accuracy, because five learned profiles preserve substantially more absolute-position flexibility than the failed four-profile design.

INTENDED_EDIT: Retain five learned mean-free positional profiles, replace the final lookup profile with a fixed normalized cosine and one learned amplitude, and initialize that amplitude to the discarded profile’s RMS scale.

EVIDENCE: Replacing two learned profiles simultaneously reduced accuracy to 74.41%, while all six learned profiles reached 99.98%; isolating one replacement tests whether five learned profiles are sufficient and removes only half as many parameters as the failed change.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with all position-common modes removed."""
=======
class MeanFreePositionEmbedding(nn.Module):
    """Five learned positional profiles and one scaled mean-free cosine."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fifth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
=======
        self.fifth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(()))
        positions = torch.arange(num_embeddings, dtype=torch.float32)
        fixed_last_profile = torch.cos(
            2.0 * math.pi * positions / num_embeddings
        )
        fixed_last_profile.sub_(fixed_last_profile.mean())
        fixed_last_profile.div_(fixed_last_profile.square().mean().sqrt())
        self.register_buffer(
            "fixed_last_profile", fixed_last_profile, persistent=False
        )
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
=======
        last_weight = (
            self.last_coordinate * self.fixed_last_profile
        ).unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
=======
                module.last_coordinate.copy_(
                    projected[:, -1].square().mean().sqrt()
                )
>>>>>>> REPLACE