# Torchfire
Firedrake provides a high-level library for the automated solution of PDE by finite element methods. In order to use Firedrake as a basis for scientific machine learning, we need to make it callable within a learning framework sy as PyTorch. This package provides such bindings by means of extending PyTorch.Function, the base class for user-defined operations.  

# Numerical problems
## Problem 1: Solving heat equation with nFEM-TorchFire

<p align="center">
<img width="800" height = "250" src="demos/heat_eq_conductivity/results/animations.gif">
<figcaption><b>Figure 1:</b> (Left) Ground truth conductivity diffusion $\kappa$, (Middle) Temperature field obtained by Firedrake software, (Right) Temperature field obtained by nFEM, TorchFire</figcaption>
</p>


## Problem 2: Conductivity diffusion field inversion from sparse observations with TNet-TorchFire
<p align="center">
<img width="550" height = "250" src="demos/tnet_heat_equation/results/animations.gif">
<figcaption><b>Figure 2:</b> (Left) Ground truth conductivity diffusion <img src="https://i.upmath.me/svg/%5Ckappa" alt="\kappa" />, (Right) Predicted Conductivity diffusion <img src="https://i.upmath.me/svg/%5Ckappa" alt="\kappa" /> obtained by TNet-TorchFire, reconstructed from 10 random observables over the domain</figcaption>
</p>


