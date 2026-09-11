MECHANISM: Identity-plus-low-rank recurrent retention gate

HYPOTHESIS: A rank-16 recurrent correction plus direct per-channel hidden feedback will restore enough state-conditioned gating to achieve at least 85% accuracy on the verified 21-step schedule while reducing total inference MACs from 275.93M to approximately 210.20M.

INTENDED_EDIT: Keep the full-rank recurrent proposal and signed-extrema readout, but replace the full-rank retention transform with an input projection, identity hidden feedback, and a learned rank-16 recurrent correction.

EVIDENCE: The input-only retention gate reduced MACs substantially but reached only 83.31%, indicating that recurrent gate conditioning matters; meanwhile, eliminating full-rank proposal interaction also failed. This design preserves the full-rank proposal while restoring recurrent retention conditioning at much lower structural cost.

<<<<<<< SEARCH
        self.hidden_size = 80
        self.input_norm = nn.LayerNorm(20)
        self.transition = nn.Linear(20 + self.hidden_size, 2 * self.hidden_size)
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(4 * self.hidden_size, 8)
        with torch.no_grad():
            self.transition.bias[: self.hidden_size].fill_(1.0)
=======
        self.hidden_size = 80
        self.retention_rank = 16
        self.input_norm = nn.LayerNorm(20)
        self.proposal = nn.Linear(20 + self.hidden_size, self.hidden_size)
        self.retention_input = nn.Linear(20, self.hidden_size)
        self.retention_reduce = nn.Linear(
            self.hidden_size, self.retention_rank, bias=False
        )
        self.retention_expand = nn.Linear(
            self.retention_rank, self.hidden_size, bias=False
        )
        self.proposal_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(4 * self.hidden_size, 8)
        with torch.no_grad():
            self.retention_input.bias.fill_(1.0)
            self.retention_input.weight.mul_(math.sqrt(0.2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, peak, trough, count = state
        update = self.transition(
            torch.cat((self.input_norm(frame), hidden), dim=-1)
        )
        retention_logits, proposal_logits = update.chunk(2, dim=-1)
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
=======
        hidden, summary, peak, trough, count = state
        normalized_frame = self.input_norm(frame)
        proposal_logits = self.proposal(
            torch.cat((normalized_frame, hidden), dim=-1)
        )
        retention_logits = (
            self.retention_input(normalized_frame)
            + hidden
            + self.retention_expand(
                torch.tanh(self.retention_reduce(hidden))
            )
        )
        retention = torch.sigmoid(retention_logits)
        proposal = torch.tanh(self.proposal_norm(proposal_logits))
>>>>>>> REPLACE