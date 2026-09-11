MECHANISM: Bias-free attention output projection

HYPOTHESIS: Removing the eight-parameter attention output bias will reduce the model from 1,384 to 1,376 parameters while retaining at least 99% accuracy within the proven 52,000-step schedule.

INTENDED_EDIT: Disable only the bias of the standard attention output linear layer, preserving the terminal LayerNorm, one-neuron MLP, training budget, and fast built-in operators.

EVIDENCE: The current model achieved 99.99% accuracy at 52,000 steps. Previous 1,376-parameter attempts changed the terminal LayerNorm and timed out, so removing an attention projection bias instead provides an informative eight-parameter reduction without the implicated LayerNorm path or added reconstruction operations.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = nn.Linear(d_model, d_model, bias=False)
>>>>>>> REPLACE