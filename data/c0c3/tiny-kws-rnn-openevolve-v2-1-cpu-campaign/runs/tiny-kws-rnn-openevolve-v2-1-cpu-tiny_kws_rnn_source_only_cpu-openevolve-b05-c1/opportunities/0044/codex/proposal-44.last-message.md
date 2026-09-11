MECHANISM: Signed-extrema temporal readout

HYPOTHESIS: An 80-unit recurrence augmented with a negative-extrema summary will achieve at least 85% validation accuracy while using approximately 315.05M total inference MACs, about 2.0% fewer than the passing 81-unit model.

INTENDED_EDIT: Reduce the recurrent width from 81 to 80 and add a parameter-free running trough alongside the existing mean, positive peak, and terminal-state views, expanding only the final classifier input.

EVIDENCE: The 80-unit model reached 84.91%—one correct validation prediction short of 85%—at 314.52M MACs, while 81 units passed at 85.77%. A signed trough restores complementary temporal information for only 0.52M additional classifier MACs.

<<<<<<< SEARCH
        self.hidden_size = 81
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(3 * self.hidden_size, 8)
=======
        self.hidden_size = 80
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(4 * self.hidden_size, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        peak = torch.zeros_like(hidden)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, count
=======
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )
        summary = torch.zeros_like(hidden)
        peak = torch.zeros_like(hidden)
        trough = torch.zeros_like(hidden)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return hidden, summary, peak, trough, count
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, summary, peak, count = state
        update = self.transition(
            torch.cat((self.input_norm(frame), hidden), dim=-1)
        )
        retention_logits, proposal_logits = update.chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
        hidden = retention * hidden + (1.0 - retention) * proposal
        return (
            hidden,
            summary + hidden,
            torch.maximum(peak, hidden),
            count + 1.0,
        )
=======
    def recurrent_step(
        self,
        frame: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden, summary, peak, trough, count = state
        update = self.transition(
            torch.cat((self.input_norm(frame), hidden), dim=-1)
        )
        retention_logits, proposal_logits = update.chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
        hidden = retention * hidden + (1.0 - retention) * proposal
        return (
            hidden,
            summary + hidden,
            torch.maximum(peak, hidden),
            torch.minimum(trough, hidden),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
=======
    def recurrent_sequence(
        self,
        frames: torch.Tensor,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
>>>>>>> REPLACE

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden, summary, peak, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(torch.cat((mean_output, peak, hidden), dim=-1))
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        hidden, summary, peak, trough, count = state
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((mean_output, peak, trough, hidden), dim=-1)
        )
>>>>>>> REPLACE