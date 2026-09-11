MECHANISM: Zero-floor cosine settling

HYPOTHESIS: Extending the successful learning-rate-floor reduction from 2% to zero will exceed 9,243 correct predictions by further suppressing late-training parameter noise without increasing runtime.

INTENDED_EDIT: Replace the 2% terminal learning-rate floor with a pure cosine decay to zero.

EVIDENCE: Lowering the floor from 10% to 2% improved validation correctness from 9,232 to 9,243 at the same batch size, architecture, and example budget, supporting a compute-neutral test of the zero-floor boundary.

<<<<<<< SEARCH
    multiplier = 0.02 + 0.98 * cosine
=======
    multiplier = cosine
>>>>>>> REPLACE