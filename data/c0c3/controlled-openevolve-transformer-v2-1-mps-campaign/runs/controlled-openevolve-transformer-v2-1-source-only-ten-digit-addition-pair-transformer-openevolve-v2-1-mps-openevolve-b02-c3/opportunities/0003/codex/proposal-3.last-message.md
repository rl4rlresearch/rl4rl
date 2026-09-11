MECHANISM: LayerNorm-absorbed query offset

HYPOTHESIS: Removing all QKV biases will reduce parameters from 1644 to 1620 while retaining at least 99% accuracy, because the remaining query bias is representable through the pre-attention LayerNorm bias and query projection; induced key and value offsets are redundant.

INTENDED_EDIT: Make the combined QKV projection bias-free.

EVIDENCE: The query-only-bias reference achieved 100% accuracy with 1628 parameters, establishing that key and value biases are unnecessary and motivating the isolated removal of the remaining 8-parameter query bias.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        # The pre-attention LayerNorm bias can supply a query offset through
        # the query projection; induced key and value offsets are redundant.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE