MECHANISM: High-frequency band pruning with further margin sharpening

HYPOTHESIS: Using 19 mel bands with label smoothing reduced to 0.01 will recover at least one validation decision, exceed 85% accuracy, and retain the measured reduction to approximately 699.7M total MACs.

INTENDED_EDIT: Remove the highest mel band from both recurrent paths, structurally reduce the GRU input width to 19, and lower label smoothing from 0.03 to 0.01.

EVIDENCE: The 19-band model reached 84.91%, one validation example short of 85%; reducing smoothing from 0.03 to 0.02 improved cross-entropy from 0.4759 to 0.4718 without changing accuracy, motivating one further margin-sharpening step.

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
            self.input_norm(frame[:, :19]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[:, :, :19]), hidden.transpose(0, 1).contiguous()
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(logits, labels, label_smoothing=0.03)
=======
    return F.cross_entropy(logits, labels, label_smoothing=0.01)
>>>>>>> REPLACE