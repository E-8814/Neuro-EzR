"""
Aligned training and inference pipeline for Neural E-Z Reader.

The differentiable EZR (training) and discrete simulation (inference) share
parameter semantics, so learned values transfer natively.

Pipeline:
    Train:     LLaMA -> L1/L2/skip -> DiffEZReader -> loss vs human data
    Inference: LLaMA -> L1/L2/skip -> Discrete EZR Simulation -> predictions
"""
