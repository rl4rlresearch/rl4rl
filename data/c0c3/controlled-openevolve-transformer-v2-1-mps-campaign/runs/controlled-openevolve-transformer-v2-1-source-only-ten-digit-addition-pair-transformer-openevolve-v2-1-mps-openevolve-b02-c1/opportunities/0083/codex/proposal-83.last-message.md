MECHANISM: Zero-offset scalar MLP

HYPOTHESIS: Fixing the width-one MLP’s hidden bias at zero will reduce the model from 1,000 to 999 parameters while retaining at least 99% accuracy, because it preserves the learned nonlinear MLP weights and all demonstrated load-bearing rank-two and relative-position attention pathways.

INTENDED_EDIT: Remove the single trainable `fc1` bias while consuming its constructor-equivalent random draw to preserve initialization of subsequent parameters.

EVIDENCE: The current model achieves 99.98% at 1,000 parameters; prior failures identify attention rank and relative-position structure as sensitive, so the previously untested scalar MLP offset is a more informative compression target while those pathways remain unchanged.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
=======
        self.fc1 = nn.Linear(d_model - 1, d_ff, bias=False)

        # Preserve the RNG stream of the removed fc1 bias.
        fc1_bias_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(
            -fc1_bias_bound, fc1_bias_bound
        )

        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
>>>>>>> REPLACE