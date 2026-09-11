MECHANISM: Continued MLP capacity scaling

HYPOTHESIS: A 5.625× MLP will retain at least 435M-token throughput and reduce val_bpb below 0.982905.

INTENDED_EDIT: Increase every MLP hidden width from 5× to 5.625× while retaining the proven batch size, architecture, and 93.75% learning rates.

EVIDENCE: Increasing MLP width from 4.375× to 5× improved val_bpb from 0.984614 to 0.982905 despite reducing token exposure from 466.6M to 451.9M, so another equal 0.625× capacity increment directly tests whether the favorable capacity-throughput tradeoff continues.

<<<<<<< SEARCH
        hidden_dim = 5 * config.n_embd
=======
        hidden_dim = 45 * config.n_embd // 8
>>>>>>> REPLACE