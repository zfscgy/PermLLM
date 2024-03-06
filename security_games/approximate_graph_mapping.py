from re import M
from typing import Callable


import numpy as np
import torch 
import tqdm

from desi_llm.common.utils import random_orthonormal


class QPCPSover:
    """Quadratic Programming with Orthogonal Constraint"""
    def __init__(self, solution_size: int, 
                 objective: Callable[[torch.Tensor], torch.Tensor], objective_grad: Callable[[torch.Tensor], torch.Tensor],
                 learning_rate: float, momentum_coef: float, q: float = 0.5, cayley_steps: int = 2,
                 device: str="cuda") -> None:
        self.solution_size = solution_size
        self.objective = objective
        self.objective_grad = objective_grad
        self.learning_rate = learning_rate
        self.momentum_coef = momentum_coef
        self.q = q
        self.cayley_steps = cayley_steps

        self.identity = torch.eye(solution_size, dtype=torch.float, device=device)
        self.X = torch.eye(solution_size, dtype=torch.float, device=device)
        self.M = 0



    def step(self):
        raw_gradient = self.objective_grad(self.X)
        self.M = self.momentum_coef * self.M - raw_gradient
        W0 = self.M @ self.X.T - 0.5 * self.X @ (self.X.T @ self.M @ self.X.T)
        W = W0 - W0.T
        self.M = W @ self.M
        step_size = min(self.learning_rate, 2 * self.q / (torch.sqrt(torch.sum(torch.square(W))) + 1e-8))
        Y = self.X + step_size * self.M
        for _ in range(self.cayley_steps):
            Y = self.X + 0.5 * step_size * W @ (self.X + Y)
        self.X = Y



    def optimize(self, stop_criterion: float = 1e-7, n_steps: int = 10000):
        objective_values = []
        for i in tqdm.tqdm(range(n_steps)):
            objective_values.append(self.objective(self.X).item())
            if i % 10000 == 0:
                self.learning_rate *= 0.9
                print(f"Current objective value: {objective_values[-1]}")
            # if len(objective_values) >= 2 and objective_values[-2] - objective_values[-1] < stop_criterion:
            #     break
            self.step()
        return self.X


if __name__ == "__main__":
    device = "cuda"
    def test_random_orthonormal():
        dimension = 1000
        X0 = random_orthonormal(dimension, device=device)
        A0 = torch.normal(0, 1, [dimension, dimension], device=device)
        A1 = torch.normal(0, 1, [dimension, dimension], device=device)
        A = A0 @ A1 
        A /= dimension ** 0.5

        A_noisy = (A0 + torch.normal(0, 0.01, [dimension, dimension], device=device)) @ (A1 + torch.normal(0, 0.01, [dimension, dimension], device=device))
        A_noisy /= dimension ** 0.5

        # A = A @ A.T
        B = X0 @ A @ X0.T

        objective = lambda X: - torch.mean((X @ A @ X.T) * B)
        # objective_grad = lambda X: - (A @ X.T @ B + A.T @ X.T @ B.T)

        def objective_grad_torch(X: torch.Tensor):
            X = X.clone()
            X.requires_grad = True
            return torch.autograd.grad(objective(X), X)[0].detach()
        objective_grad = objective_grad_torch


        solver = QPCPSover(dimension, objective, objective_grad, 0.1, 0.9, device=device)

        X1 = solver.optimize(n_steps=100_0000)
        print(X1.T @ X1)
        print(f"Relative error: {(torch.sqrt(torch.mean(torch.square(X1 - X0))) / torch.std(X0)).item():.4f}")
        print(f"A * B Ratio: {(torch.sum((X1 @ A @ X1.T) * B) / torch.sum(B * B)).item():.4f}")
        print(f"|A - B| Error: {(torch.sqrt(torch.mean(torch.square(B - X1 @ A @ X1.T))) / torch.sqrt(torch.mean(torch.square(B)))).item():.4f}")

        print(X1)
        print(X0)

    def test_graph_matching():
        dimension = 1000
        A0 = torch.normal(0, 1, [dimension, dimension], device=device)
        A1 = torch.normal(0, 1, [dimension, dimension], device=device)
        A = A0 @ A1 
        A /= dimension ** 0.5

        A_noisy = (A0 + torch.normal(0, 0.01, [dimension, dimension], device=device)) @ (A1 + torch.normal(0, 0.01, [dimension, dimension], device=device))
        A_noisy /= dimension ** 0.5
        

        raw_permutation = torch.randperm(dimension, device=device)
        B = A[raw_permutation][:, raw_permutation]

        objective = lambda X: - torch.mean((X @ A_noisy @ X.T) * B)
        # objective_grad = lambda X: - (A @ X.T @ B + A.T @ X.T @ B.T)

        def objective_grad_torch(X: torch.Tensor):
            X = X.clone()
            X.requires_grad = True
            return torch.autograd.grad(objective(X), X)[0].detach()
        objective_grad = objective_grad_torch

        solver = QPCPSover(dimension, objective, objective_grad, 0.1, 0.9, device=device)
        P1 = solver.optimize(n_steps=100_0000)
        recovered_permutation = torch.argmax(P1, dim=1)

        print(f"Indices accuracy: {torch.mean((raw_permutation == recovered_permutation).float()).item():.4f}")

    # test_random_orthonormal()
    test_graph_matching()