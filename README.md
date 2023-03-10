# Learning-NN-ROA

Codes in this repository implement a cutting-plane method to synthesize Lyapunov functions for uncertain neural network dynamical systems. The details of the method and the problem setup can be found in our paper:

[Learning Region of Attraction for Nonlinear Systems](https://arxiv.org/abs/2110.00731)\
Shaoru Chen, Mahyar Fazlyab, Manfred Morari, George J. Pappas, Victor M. Preciado.\
IEEE Conference on Decision and Control (CDC), 2021.

## Methodology

### Abstraction through (uncertain) NN dynamics
We consider finding an inner estimation of the region of attraction (ROA) of a discrete-time nonlinear system $x_+ = f(x)$ through Lyapunov functions. Since analysis directly based on the general nonlinear function $f(x)$ is hard to execute, we propose to train a NN $f_{NN}(x)$ to approximate $f(x)$ and bound the approximation error in a form amenable for analysis. Specifically, we construct an uncertain NN dynamical system as follows:

$x_+ = Ax + f_{NN}(x) + w(x)$ with $\lVert w(x) \rVert_\infty \leq \gamma \lVert x \rVert_\infty + \delta, \forall x \in \mathcal{X}$

which over-approximates the dynamics $x_+ = f(x)$ over a compact region of interest $\mathcal{X} \subset \mathbb{R}^n$. It follows that any robust ROA innner approximation of the uncertain NN system is an inner appproximation of the ROA of the original nonlinear system. 

In this way, we can benchmark the ROA analysis of general nonlinear dynamics on uncertain NN dynamical systems. Next, we only need to focus on developing efficient numerical tools for (uncertain) NN dynamical systems. 

### Cutting-plane method for Lyapunov function synthesis
We parameterize the robust Lyapunov function for the uncertain NN dynamics as a quadratic function $V(x) = z(x)^\top P z(x)$ with nonlinear basis $z(x)$. In this way, the set of valid robust Lyapunov functions is *convex* in the parameter space of $P$. Then, we can apply the [cutting-plane method](https://see.stanford.edu/materials/lsocoee364b/05-localization_methods_notes.pdf) to find a feasible $P$ (i.e., a valid robust Lyapunov function) with finite-step termination guarantees. This algorithm can be implemented in a counterexample-guided synthesis fashion and iterates between calling a learner (solving a finite-dimensional convex program) and a verifier (solving a non-convex mixed-integer quadratic program, handled by [gurobi](https://cdn.gurobi.com/wp-content/uploads/quadratic_optimization.pdf) ver 9.0+).

### Demonstration
Consider the rational system 
<p float="right">
<img src="https://github.com/ShaoruChen/web-materials/blob/main/ROA_approx_CDC_21/rational_system.png" width="500" height="60">
</p>

By simulating several trajectories, we can roughly guess the area of the ROA and fix the region of interest $\mathcal{X}$ as denoted by the red dashed line below. Our algorithm then finds an inner approximation of the ROA for the rational system as denoted by the blue lines in the following figure. 

<p float="right">
<img src="https://github.com/ShaoruChen/web-materials/blob/main/ROA_approx_CDC_21/rational_ROA.png" width="350" height="350">
</p>

## Running the experiments
You can run the examples in the [examples](https://github.com/ShaoruChen/Learning-NN-ROA/tree/main/examples) folder to recover the simulation in the paper. 

Required packages:
- [pympc](https://github.com/TobiaMarcucci/pympc) by Tobia Marcucci (codes included in this repo.)
- Keras
- gurobipy (license required to run the gurobi solver)
