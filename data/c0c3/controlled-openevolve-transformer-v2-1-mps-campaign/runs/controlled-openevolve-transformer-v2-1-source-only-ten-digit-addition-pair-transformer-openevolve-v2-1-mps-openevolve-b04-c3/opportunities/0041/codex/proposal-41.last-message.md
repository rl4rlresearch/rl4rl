MECHANISM: MLP hidden-bias sharing with time-compressed optimization

HYPOTHESIS: Sharing one hidden-unit bias will reduce the qualified model to 1,395 parameters, while 45,000 steps at 5e-3—approximately preserving the qualified 75,000-step schedule’s cumulative learning rate—will achieve at least 99% accuracy and finish verification.

INTENDED_EDIT: Tie the first two MLP hidden biases, compress training to 45,000 higher-learning-rate steps, and validate only at the final positive step.

EVIDENCE: The current 1,396-parameter model achieved 99.97%, but 60,000–75,000-step 1,395-parameter trials repeatedly timed out and completed 45,000-step trials at the original learning rate scored 0%; this motivates testing a less attention-sensitive one-parameter constraint with a time-compressed optimizer schedule.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        if d_ff < 2:
            raise ValueError("d_ff must be at least 2")
        self.fc1 = nn.Linear(d_model, d_ff)
        # Hidden units retain independent input and output weights while the
        # first two share their scalar activation offset.
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_bias = torch.cat(
            (self.fc1.bias[:1], self.fc1.bias[:1], self.fc1.bias[1:])
        )
        hidden = F.linear(x, self.fc1.weight, hidden_bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (step > 0 and step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=75000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=75000)
=======
    p.add_argument("--train-steps", type=int, default=45000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=45000)
>>>>>>> REPLACE