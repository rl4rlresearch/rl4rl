MECHANISM: Token-horizon-normalized NorMuon variance smoothing

HYPOTHESIS: Raising Muon beta2 from 0.95 to 0.965 will compensate for the 176K batch’s 45% higher update frequency, reduce noisy matrix updates, and lower val_bpb below 0.984418.

INTENDED_EDIT: Increase only NorMuon’s second-moment decay to approximately `0.95 ** (176/256)`, preserving all other architecture, schedule, and optimizer settings.

EVIDENCE: Normalizing step-dependent weight decay for the 176K regime improved val_bpb from 0.986967 to 0.984418, while the 160K batch regression suggests gradient noise limits further update-frequency gains; this motivates normalizing another step-based Muon timescale.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=5, beta2=0.965, weight_decay=weight_decay,
>>>>>>> REPLACE