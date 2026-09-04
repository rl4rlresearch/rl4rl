MECHANISM: Six-terminal relative-kernel tie with sparse validation

HYPOTHESIS: Extending the qualified five-terminal tie by one will reduce the model from 1,116 to 1,115 parameters while retaining at least 99% accuracy; reducing validation frequency will allow all 4,999 training updates to finish within the verification limit.

INTENDED_EDIT: Shorten the learned relative-bias vector by one, reconstruct six tied terminal logits, and evaluate every 1,000 steps while retaining mandatory final evaluation and checkpointing.

EVIDENCE: The five-terminal design achieved 100% accuracy at 1,116 parameters, whereas projection-side 1,116-parameter reductions collapsed. The earlier six-terminal attempt timed out without producing contrary accuracy evidence.

<<<<<<< SEARCH
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the five terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 5)
        )
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the six terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 6)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias,
                self.relative_bias.new_zeros(5),
=======
                self.relative_bias,
                self.relative_bias.new_zeros(6),
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE