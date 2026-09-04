MECHANISM: Adjacent trailing-frame pruning on the qualified spectral trunk

HYPOTHESIS: The qualified 84-unit GRU using mel bands 1–17 will retain at least 85% validation accuracy when reduced from 28 to 27 causal steps, lowering total inference MACs from 581,910,000 to approximately 561,166,620.

INTENDED_EDIT: Adopt the qualified 84-unit, 17-band architecture and omit one additional trailing frame while preserving its dual-view classifier and training procedure.

EVIDENCE: The 28-step, 84-unit, bands-1–17 design achieved 86.87% validation accuracy at 581,910,000 MACs, leaving a 1.87-point margin; the closest unresolved reduction is one trailing recurrent step, saving approximately 20,743,380 MACs without reducing spectral or hidden capacity.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
=======
        self.input_norm = nn.LayerNorm(17)
        self.gru = nn.GRU(17, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[..., 1:18]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., 1:18]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(1, available_frames - 2))
=======
        return list(range(1, available_frames - 4))
>>>>>>> REPLACE