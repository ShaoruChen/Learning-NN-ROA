# Learning-NN-ROA

Codes in this repo. implement a cutting-plane method to synthesize Lyapunov functions for uncertain neural network dynamical systems. The details of the method and the problem setup can be found in our paper:

Shaoru Chen, Mahyar Fazlyab, Manfred Morari, George J. Pappas, Victor M. Preciado. [Learning Region of Attraction for Nonlinear Systems](https://arxiv.org/abs/2110.00731), IEEE Conference on Decision and Control (CDC), 2021.

You can run the examples in the [examples](https://github.com/ShaoruChen/Learning-NN-ROA/tree/main/examples) folder to recover the simulation in the paper. 

Required packages:
- [pympc](https://github.com/TobiaMarcucci/pympc) by Tobia Marcucci (codes included in this repo.)
- Keras
- gurobipy (license required to run the gurobi solver)
