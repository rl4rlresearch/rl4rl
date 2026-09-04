MECHANISM: Six-terminal relative-kernel tie with final-only validation

HYPOTHESIS: Tying six terminal kernel logits will reduce the verified nested-rotation design from 1,117 to 1,115 parameters while retaining at least 99% accuracy, and eliminating interim validation will allow all 4,999 updates to finish within the verification limit.

INTENDED_EDIT: Reconstruct six tied terminal relative-bias logits instead of four and evaluate only after the final training update, which still writes both required checkpoints.

EVIDENCE: The five-terminal design achieved 100% accuracy at 1,116 parameters; six-terminal attempts timed out rather than showing an accuracy failure, including with validation every 1,000 steps, so removing all nonessential interim evaluations directly tests the remaining runtime confound.

<<<<<<< SEARCH
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the four terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 4)
        )
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the six terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 6)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(4),
            )
        )
=======
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(6),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE