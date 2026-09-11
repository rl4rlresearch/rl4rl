MECHANISM: Attention-output gauge reuse

HYPOTHESIS: Reusing the functionally redundant common-mode attention projection bias as the learned query offset will reduce the model from 1,377 to 1,376 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Remove the standalone query-offset parameter and derive its zero-initialized learned value from the mean attention projection bias.

EVIDENCE: The 1,377-parameter model achieved 100% accuracy at 50,000 steps, whereas fixing the offset to one changed its successful zero-initialized optimization path and timed out. The projection bias’s common-mode component is erased by downstream LayerNorms, so it can encode the query offset without reducing effective model capacity.

<<<<<<< SEARCH
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias.view(1, 1, 1, 1)
=======
        q = q + self.proj.bias.mean()
>>>>>>> REPLACE