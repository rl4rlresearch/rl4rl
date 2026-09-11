MECHANISM: Further-shortened AdamW refinement tail

HYPOTHESIS: Pairing the verified 59% Muon cooldown with a 30% AdamW cooldown will beat val_bpb 0.985708 by extending the productive late embedding and unembedding updates.

INTENDED_EDIT: Shorten only AdamW’s linear cooldown from 35% to 30%, retaining Muon’s best verified 59% cooldown and all other settings.

EVIDENCE: With Muon fixed at 59%, shortening AdamW cooldown from 45% to 40% improved val_bpb from 0.986100 to 0.985875, and shortening it again to 35% improved val_bpb to 0.985708; testing 30% directly probes whether that trend continues.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.35   # further shorten AdamW cooldown to preserve late refinement
=======
WARMDOWN_RATIO = 0.30   # extend productive late AdamW embedding and unembedding updates
>>>>>>> REPLACE