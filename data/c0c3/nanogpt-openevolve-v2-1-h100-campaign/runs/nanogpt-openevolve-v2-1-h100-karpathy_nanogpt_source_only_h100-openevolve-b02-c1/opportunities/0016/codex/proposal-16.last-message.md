MECHANISM: Faster channel-wise lexical adaptation

HYPOTHESIS: Increasing only the channel-wise lexical expert’s Adam learning rate from 0.005 to 0.02 will reduce val_bpb below 0.994364 without affecting throughput.

INTENDED_EDIT: Separate `bigram_lambdas` from the residual-scalar optimizer group and train them at four times their current learning rate while preserving all other parameters and schedules.

EVIDENCE: Channel-wise lexical scaling improved val_bpb from 0.995511 to 0.994364, while added gating, embeddings, and multi-depth inputs regressed; this isolates optimization of the successful mechanism without adding computation.

<<<<<<< SEARCH
        resid_params = [self.resid_lambdas, self.bigram_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
=======
        resid_params = [self.resid_lambdas]
        bigram_params = [self.bigram_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(bigram_params) + len(x0_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=bigram_params, lr=scalar_lr * 0.04, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE