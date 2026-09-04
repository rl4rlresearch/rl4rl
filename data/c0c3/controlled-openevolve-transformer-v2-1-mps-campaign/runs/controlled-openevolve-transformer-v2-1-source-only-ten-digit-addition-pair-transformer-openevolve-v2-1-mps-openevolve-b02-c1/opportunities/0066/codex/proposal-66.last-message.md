MECHANISM: Bias-free scalar MLP gate

HYPOTHESIS: Removing the single hidden-layer MLP bias will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because prior removal of the larger seven-parameter MLP residual bias retained 99.95%, whereas further compression of positional and final-normalization pathways failed.

INTENDED_EDIT: Make the one-unit MLP’s first projection bias-free while explicitly consuming the removed constructor’s bias initialization draw to preserve the successful initialization stream.

EVIDENCE: Removing the seven-parameter MLP residual bias retained 99.95% accuracy, while head-0 positional sharing, head-1 quintuplet sharing, and adjacent final-LayerNorm bias sharing all fell below threshold; this makes the remaining scalar MLP bias the best-supported one-parameter ablation outside load-bearing pathways.

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
=======
        self.fc1 = nn.Linear(d_model - 1, d_ff, bias=False)

        # Preserve the RNG stream of the removed hidden-bias initialization.
        fc1_bound = 1.0 / math.sqrt(d_model - 1)
        torch.empty(d_ff).uniform_(-fc1_bound, fc1_bound)

        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
>>>>>>> REPLACE