MECHANISM: Evenly spaced global-context injection

HYPOTHESIS: Full attention at layers 3, 6, and 8 will beat 0.984244 val_bpb by giving global information more downstream processing while retaining the proven layer-6 and final-layer full attention at unchanged compute cost.

INTENDED_EDIT: Move the earliest full-attention layer from depth 4 to depth 3, preserving three full-attention layers and every other setting.

EVIDENCE: Layers 4, 6, and 8 achieved the best 0.984244 val_bpb; moving the layer-6 attention to layers 5 or 7 regressed, establishing layer 6 as valuable and motivating an isolated placement test of the remaining non-final full-attention layer.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 4, 6, and 8
=======
WINDOW_PATTERN = "SSLSSLSL" # full attention at layers 3, 6, and 8
>>>>>>> REPLACE