MECHANISM: Softmax-equivalent rank-7 classifier factorization

HYPOTHESIS: The qualified 84-unit, 18-band GRU with a linear 168→7→8 classifier will retain at least 85% validation accuracy while reducing total inference MACs from 608,609,400 to approximately 608,518,120.

INTENDED_EDIT: Use the qualified lowest 18 mel bands and replace the 168→8 classifier with two linear layers having a seven-dimensional bottleneck and no intervening nonlinearity.

EVIDENCE: The 18-band trunk achieved 85.03% accuracy at 608,609,400 MACs. For eight-class softmax, seven independent logit contrasts suffice, so a rank-7 linear head preserves the necessary classifier output dimension while structurally saving 112 MACs per example; unlike the failed averaging head, it retains both recurrent views independently.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
=======
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(168, 7),
            nn.Linear(7, 8),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame[..., :19]).unsqueeze(1),
=======
            self.input_norm(frame[..., :18]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames[..., :19]), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., :18]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE