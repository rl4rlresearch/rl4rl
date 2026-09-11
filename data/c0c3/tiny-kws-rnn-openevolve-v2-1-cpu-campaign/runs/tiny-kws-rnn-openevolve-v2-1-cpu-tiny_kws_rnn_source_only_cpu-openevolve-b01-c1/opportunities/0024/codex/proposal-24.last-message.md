MECHANISM: Low-frequency boundary-band pruning

HYPOTHESIS: Dropping the lowest log-mel band while preserving the verified 92-unit GRU and 28-step schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 0.9%.

INTENDED_EDIT: Change the recurrent input width from 20 to 19 and feed bands 1–19 to both recurrent execution paths, leaving temporal coverage, hidden capacity, and training unchanged.

EVIDENCE: The verified 28-step, 92-unit model achieved 85.40%, while reducing either recurrent steps or hidden width failed; pruning one frequency-boundary input feature tests a distinct structural cost axis without sacrificing the temporal coverage or recurrent capacity those failures indicate are important.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 92, num_layers=1, batch_first=True)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[:, 1:]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[:, :, 1:]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE