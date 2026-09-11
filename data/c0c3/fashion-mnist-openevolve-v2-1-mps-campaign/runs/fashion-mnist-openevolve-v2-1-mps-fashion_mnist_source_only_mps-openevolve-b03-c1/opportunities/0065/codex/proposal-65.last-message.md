MECHANISM: Learned global mean–maximum attention fusion

HYPOTHESIS: A single globally learned fusion weight will exceed 9,162 correct predictions by adapting the successful mean–maximum balance without the overfitting and instability observed with 96 independent channel weights.

INTENDED_EDIT: Add one zero-initialized scalar fusion logit and replace the fixed 50/50 gate average with its sigmoid-bounded learned mixture, increasing parameters to 249,763 while preserving the current computation exactly at initialization.

EVIDENCE: Fixed equal fusion improved mean-only attention from 9,153 to 9,162 correct, but per-channel learned fusion fell to 9,141; this motivates testing the same adaptation with one strongly shared degree of freedom.

<<<<<<< SEARCH
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.classifier = nn.Sequential(
=======
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.channel_fusion_logit = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
        )
=======
        average_weight = torch.sigmoid(self.channel_fusion_logit)
        channel_signal = (
            average_weight * self.channel_gate(average_context)
            + (1.0 - average_weight) * self.channel_gate(maximum_context)
        )
>>>>>>> REPLACE