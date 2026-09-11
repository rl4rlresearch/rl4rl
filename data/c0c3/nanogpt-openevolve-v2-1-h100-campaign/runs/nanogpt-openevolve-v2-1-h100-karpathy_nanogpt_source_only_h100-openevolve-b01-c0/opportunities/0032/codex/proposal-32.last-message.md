MECHANISM: Late-stack global-context concentration

HYPOTHESIS: Moving the third full-attention layer from depth 6 to depth 7 will beat 0.984244 val_bpb because the depth-6 addition helped while the depth-2 addition hurt, suggesting global attention is more useful later in the stack.

INTENDED_EDIT: Use full attention at layers 4, 7, and 8 while preserving compute scale, optimizer, batching, and schedule.

EVIDENCE: Full-attention layers at 4, 6, and 8 achieved 0.984244, whereas adding an early full-attention layer at depth 2 regressed to 0.985402; relocating rather than adding a layer isolates whether later placement drives the benefit.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 4, 6, and 8
=======
WINDOW_PATTERN = "SSSLSSLL" # full attention at layers 4, 7, and 8
>>>>>>> REPLACE