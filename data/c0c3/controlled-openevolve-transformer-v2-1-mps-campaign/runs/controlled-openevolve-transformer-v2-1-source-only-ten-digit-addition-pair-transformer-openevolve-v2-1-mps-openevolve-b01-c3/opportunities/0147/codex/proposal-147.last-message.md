MECHANISM: Singleton-versus-sextet attention bias with final-only validation

HYPOTHESIS: The 853-parameter transformer will achieve at least 99% accuracy because the qualified 854-parameter model reached 99.97%, while this removes only one attention output-bias degree of freedom; evaluating only after training will allow the previously time-censored architecture to complete.

INTENDED_EDIT: Merge the attention output-bias groups from `[a,b,b,c,c,c,c]` to `[a,b,b,b,b,b,b]` and run validation only at the final step.

EVIDENCE: The 854-parameter endpoint-anchored design achieved 99.97%. Both prior attempts at this isolated 853-parameter reduction timed out without accuracy evidence, including one with merely reduced validation frequency, motivating a definitive final-only evaluation.

<<<<<<< SEARCH
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.output_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (
                self.output_bias[:1],
                self.output_bias[1:2].expand(2),
                self.output_bias[-1:].expand(4),
            )
        )
=======
        bias = torch.cat(
            (
                self.output_bias[:1],
                self.output_bias[-1:].expand(6),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE