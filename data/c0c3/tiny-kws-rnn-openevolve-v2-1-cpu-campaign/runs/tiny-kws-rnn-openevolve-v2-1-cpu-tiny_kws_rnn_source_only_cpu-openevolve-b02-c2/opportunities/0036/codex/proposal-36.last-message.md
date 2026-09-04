MECHANISM: Conservative high-frequency edge-band pruning

HYPOTHESIS: An 84-unit GRU using the lowest 18 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 614,565,420 to approximately 608,609,400.

INTENDED_EDIT: Move to the qualified 84-unit architecture and structurally discard the two highest-frequency mel bands in both recurrent execution paths.

EVIDENCE: The 84-unit, 19-band model achieved 86.75% accuracy at 614,565,420 MACs, providing 1.75 percentage points of margin; removing one additional edge band preserves the demonstrated recurrent capacity while saving exactly 5,956,020 GRU MACs.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 87, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)
=======
        self.input_norm = nn.LayerNorm(18)
        self.gru = nn.GRU(18, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 87, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 87, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[..., :18]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., :18]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE