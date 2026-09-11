MECHANISM: Context-conditioned bilinear lexical readout

HYPOTHESIS: Replacing the fixed additive bigram shortcut with an initially equivalent context-conditioned lexical scale will lower val_bpb below 0.984068 by letting the full contextual state determine which current-token features should influence each prediction channel.

INTENDED_EDIT: Add a zero-initialized readout matrix that computes a bounded channel-wise lexical adjustment from the final contextual state, producing an explicit bilinear interaction between context and the current-token value embedding while preserving baseline behavior at initialization.

EVIDENCE: The direct lexical expert improved val_bpb from 0.995511 to 0.994364, but the lexical-only multiplicative trigram readout failed at 0.987766 versus 0.987386. This tests a different load-bearing assumption—fixed additive separation of lexical and contextual evidence—by making lexical influence conditional on the fully contextualized representation rather than multiplying lexical embeddings alone.

<<<<<<< SEARCH
        # Per-channel strength of the direct lexical (bigram) prediction expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        # Value embeddings
=======
        # Base strength and context-conditioned modulation of the direct lexical expert
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_embd,), 0.1))
        self.contextual_lexical_gate = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.bigram_lambdas.fill_(0.1)
        # Value embeddings
=======
        self.bigram_lambdas.fill_(0.1)
        # A zero gate exactly recovers the verified additive lexical readout initially
        torch.nn.init.zeros_(self.contextual_lexical_gate.weight)
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
=======
        transformer_matrices = (sum(p.numel() for p in self.transformer.h.parameters()) +
                                sum(p.numel() for p in self.contextual_lexical_gate.parameters()))
>>>>>>> REPLACE

<<<<<<< SEARCH
        matrix_params = list(self.transformer.h.parameters())
=======
        matrix_params = (list(self.transformer.h.parameters()) +
                         list(self.contextual_lexical_gate.parameters()))
>>>>>>> REPLACE

<<<<<<< SEARCH
        lexical = norm(ve)
        lexical = lexical * self.bigram_lambdas.view(1, 1, -1)
        x = norm(x + lexical)
=======
        context = norm(x)
        lexical = norm(ve)
        lexical_scale = (
            self.bigram_lambdas.view(1, 1, -1)
            + torch.tanh(self.contextual_lexical_gate(context))
        )
        x = norm(x + lexical * lexical_scale)
>>>>>>> REPLACE