MECHANISM: Initialization-preserving frozen positional contrast

HYPOTHESIS: Freezing only the final contrast coefficient of the sixth positional profile will reduce the model from 1,524 to 1,523 learned parameters while retaining at least 99% accuracy, because the model keeps the coefficient’s original initialized value and all other positional degrees of freedom remain trainable.

INTENDED_EDIT: Store the sixth profile’s final mean-free basis coefficient as a checkpointed nontrainable buffer while retaining its remaining coefficients as learned parameters.

EVIDENCE: Replacing the entire sixth learned positional profile with one cosine amplitude nearly passed at 98.89%; freezing just one of that profile’s coefficients is substantially less restrictive and preserves the verified model’s exact initial positional embedding.

<<<<<<< SEARCH
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
=======
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 2))
        self.register_buffer("fixed_last_contrast", torch.empty(()))
>>>>>>> REPLACE

<<<<<<< SEARCH
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
=======
        last_coefficients = torch.cat(
            (self.last_coordinate, self.fixed_last_contrast.unsqueeze(0))
        )
        last_weight = (self.position_basis @ last_coefficients).unsqueeze(1)
        last = F.embedding(idx, last_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
=======
                last_coefficients = projected[:, -1] @ module.position_basis
                module.last_coordinate.copy_(last_coefficients[:-1])
                module.fixed_last_contrast.copy_(last_coefficients[-1])
                module.removed_second_common.copy_(projected[:, 1].mean())
>>>>>>> REPLACE