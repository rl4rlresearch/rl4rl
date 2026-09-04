MECHANISM: Head-shared scalar query addressing

HYPOTHESIS: A 1,163-parameter transformer will maintain at least 99% accuracy because the verified 1,171-parameter model achieved 100% with a head-shared scalar key, while distinct relative biases and value channels can still specialize the two heads when the scalar query is also shared.

INTENDED_EDIT: Replace the two-head scalar query projection with one shared scalar query projection, relying on the existing broadcast to both heads and removing eight learned parameters.

EVIDENCE: The current 1,171-parameter design reached 100% after sharing the key projection across heads; this symmetric next reduction preserves separate learned query and key functions, full-width values, and the proven eleven-bin positional routing.

<<<<<<< SEARCH
        self.query = nn.Linear(d_model, n_head, bias=False)
=======
        self.query = nn.Linear(d_model, 1, bias=False)
>>>>>>> REPLACE