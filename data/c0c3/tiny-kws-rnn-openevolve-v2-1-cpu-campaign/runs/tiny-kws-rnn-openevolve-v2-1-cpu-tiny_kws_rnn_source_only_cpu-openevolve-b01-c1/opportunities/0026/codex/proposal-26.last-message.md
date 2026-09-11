MECHANISM: High-frequency band pruning with sharper supervision

HYPOTHESIS: Removing the highest mel band while reducing label smoothing from 0.03 to 0.02 will recover at least one validation decision and reach 85% accuracy, while retaining the previously measured reduction to approximately 699.7M total MACs.

INTENDED_EDIT: Use bands 0–18 with a 19-input GRU in both recurrent paths and slightly reduce label smoothing to strengthen class margins.

EVIDENCE: The prior high-band-pruned model reached 84.91% accuracy—just below the requirement—while improving validation cross-entropy from 0.4833 to 0.4759 and reducing total MACs from 706.0M to 699.7M, indicating that its representation remained strong enough for a small loss adjustment to plausibly cross the threshold.

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
            self.input_norm(frame[..., :19]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., :19]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
>>>>>>> REPLACE