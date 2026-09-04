MECHANISM: Input-conditioned retention with full-rank recurrent proposal

HYPOTHESIS: Making only the retention gate input-conditioned will retain at least 85% validation accuracy on the verified 21-step schedule while reducing recurrent MACs from 273.84M to approximately 164.30M and total inference MACs to approximately 166.39M.

INTENDED_EDIT: Split the transition into a full-rank recurrent proposal and a cheaper input-only retention gate, preserving the 80-channel state, signed-extrema readout, schedule, and training procedure.

EVIDENCE: The diagonal-memory design failed after eliminating full-rank hidden interaction, while the current full-rank model passes at 85.40%; this patch preserves an 80-dimensional full-rank hidden-to-proposal transform and removes only the gate’s hidden-to-hidden matrix.

<<<<<<< SEARCH
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(4 * self.hidden_size, 8)
        with torch.no_grad():
            self.transition.bias[: self.hidden_size].fill_(1.0)
=======
        self.retention = nn.Linear(20, self.hidden_size)
        self.transition = nn.Linear(20 + self.hidden_size, self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(4 * self.hidden_size, 8)
        with torch.no_grad():
            self.retention.bias.fill_(1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        update = self.transition(
            torch.cat((self.input_norm(frame), hidden), dim=-1)
        )
        retention_logits, proposal_logits = update.chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
=======
        normalized_frame = self.input_norm(frame)
        retention = torch.sigmoid(self.retention(normalized_frame))
        proposal_logits = self.transition(
            torch.cat((normalized_frame, hidden), dim=-1)
        )
>>>>>>> REPLACE