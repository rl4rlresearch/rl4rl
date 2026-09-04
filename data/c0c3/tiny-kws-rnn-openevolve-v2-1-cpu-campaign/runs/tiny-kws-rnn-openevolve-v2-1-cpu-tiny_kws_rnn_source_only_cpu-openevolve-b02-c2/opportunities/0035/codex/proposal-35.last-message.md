MECHANISM: Edge-band structural pruning while preserving recurrent capacity

HYPOTHESIS: An 84-unit GRU using the lowest 19 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 620,521,440 to approximately 614,565,420.

INTENDED_EDIT: Preserve the qualified dual-view recurrent architecture and training procedure, reduce hidden width from 86 to 84, and structurally remove the highest-frequency mel band from every recurrent input.

EVIDENCE: The 84-unit, 20-band model achieved 85.64% accuracy, whereas reducing recurrent width to 83 failed at 84.66%; preserving width 84 while pruning only one edge band targets input-side MACs without crossing the observed recurrent-capacity boundary. The failed 20-to-16 projection further favors a conservative one-band reduction over aggressive spectral compression.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[..., :19]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., :19]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE