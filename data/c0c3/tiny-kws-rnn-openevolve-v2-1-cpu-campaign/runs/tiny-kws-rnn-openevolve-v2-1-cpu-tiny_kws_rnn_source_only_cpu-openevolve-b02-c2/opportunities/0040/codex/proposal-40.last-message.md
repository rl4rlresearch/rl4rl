MECHANISM: Adjacent high-frequency edge-band pruning

HYPOTHESIS: An 84-unit GRU using the lowest 17 mel bands over the qualified 29-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 608,609,400 to approximately 602,653,380.

INTENDED_EDIT: Adopt the qualified 84-unit architecture and structurally discard the three highest-frequency mel bands in both recurrent execution paths.

EVIDENCE: The 84-unit, 19-band model achieved 86.75% accuracy and the 18-band model remained qualified at 85.03%; preserving recurrent width while removing the next edge band is the closest unresolved structural reduction and saves exactly 5,956,020 recurrent MACs.

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
            self.input_norm(frame[..., :17]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., :17]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE