# torchfire
Firedrake provides a high-level library for the automated solution of PDE by finite element methods.  
In order to use Firedrake as a basis for scientific machine learning, we need to make it callable within a learning framework sy as PyTorch.
This package provides such bindings by means of extending PyTorch.Function, the base class for user-defined operations.  
