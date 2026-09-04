MECHANISM: Learned low-rank spectral bottleneck

HYPOTHESIS: A rank-17 input projection will preserve at least 85% validation accuracy on the qualified 25-step schedule while reducing total dense inference MACs from 294,573,600 to approximately 290,498,600.

INTENDED_EDIT: Preserve the 60-unit GRU, temporal summaries, and frames 3–26 plus frame 28, but factor each GRU input transform through an orthogonally initialized bias-free 20-to-17 Linear projection.

EVIDENCE: The current 25-step schedule qualified at 85.15%, whereas further temporal truncation and disrupting early continuity failed; an orthogonal spectral-rank reduction is a conservative, independent way to reduce matrix cost without removing another observed frame.

<<<<<<< SEARCH
        self.hidden_size = 60
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
=======
        self.hidden_size = 60
        self.spectral_size = 17
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(
            20, self.spectral_size, bias=False
        )
        nn.init.orthogonal_(self.input_projection.weight)
        self.gru = nn.GRU(
            self.spectral_size,
            self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        hidden, summary, maximum, count = state
        encoded = self.input_projection(self.input_norm(frame))
        encoded = encoded * math.sqrt(20.0 / self.spectral_size)
        output, hidden = self.gru(
            encoded.unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        hidden, summary, maximum, count = state
        encoded = self.input_projection(self.input_norm(frames))
        encoded = encoded * math.sqrt(20.0 / self.spectral_size)
        outputs, hidden = self.gru(
            encoded, hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE