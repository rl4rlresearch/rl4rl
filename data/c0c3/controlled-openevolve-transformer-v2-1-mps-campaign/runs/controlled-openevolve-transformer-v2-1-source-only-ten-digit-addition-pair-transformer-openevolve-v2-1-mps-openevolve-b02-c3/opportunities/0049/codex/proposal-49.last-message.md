MECHANISM: Second final-classifier latent-bias coordinate fixing

HYPOTHESIS: Fixing a second zero-initialized final latent-bias coordinate at zero will reduce the model from 1,033 to 1,032 parameters while retaining at least 99% accuracy, because the remaining three learned latent-bias coordinates, trainable token codes, and upstream projections preserve sufficient output calibration.

INTENDED_EDIT: Shorten the learned final latent bias from four coordinates to three and reconstruct the final two coordinates as fixed zeros.

EVIDENCE: The immediately preceding one-coordinate final-bias fix achieved 99.77% accuracy with 1,033 parameters, whereas similarly sized reductions in attention routing, key scaling, and query bias collapsed; extending the successful output-path reduction by one coordinate is the narrowest informative next test.

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 1)
        )
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        final_bias = F.pad(self.final_bias, (0, 1))
=======
        final_bias = F.pad(self.final_bias, (0, 2))
>>>>>>> REPLACE