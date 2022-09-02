"""Knapsack solvers for the well known Knapsack Problem.

This module shows different implementations of Knapsack solvers.
So far implemented:

    - Brand-and-Bound with capacity constraint relaxation.
    - Brand-and-Bound with integrality constraint relaxation.

You can do an example run by, for example, ::
    $value, taken = BranchBoundIntegralityConstraint(items, capacity).execute()

"""

from typing import List, Tuple, Union

import numpy as np

from algorithms.branch_and_bound import BranchAndBoundSolver, Item, Solution


class DepthFirstSolver(BranchAndBoundSolver):
    def _calculate_optimistic_estimate(self, solution: Solution) -> Union[int, float]:
        pass

    def execute(self) -> Tuple[int, List[int]]:
        """Execute branch and bound algorithm."""
        # Define initial solution and save in queue
        upperbound = self._calculate_optimistic_estimate(Solution(0, 0, 0, 0, []))
        init_solution = Solution(0, 0, 0, upperbound, [])
        self.queue.append(init_solution)

        while self.queue:
            solution = (
                self.queue.pop()
            )  # Get the solution that last entered the queue (=depth first strategy)
            self._explore_tree(solution=solution)  # Explore solution

        return self._get_final_solution()


class BranchBoundCapacityConstraint(DepthFirstSolver):
    def __init__(self, items: List[Item], capacity: int) -> None:
        super().__init__(items, capacity)
        self.cumsum_value = np.insert(np.cumsum([value for value in self.values]), 0, 0)

    def _calculate_optimistic_estimate(self, solution: Solution) -> int:
        """Calculate upperbound by relaxing capacity constraint."""
        next_item_idx = solution.level
        estimate: int = (
            solution.weight + self.cumsum_value[-1] - self.cumsum_value[next_item_idx]
        )
        return estimate


class BranchBoundIntegralityConstraint(DepthFirstSolver):
    def _calculate_optimistic_estimate(self, solution: Solution) -> Union[int, float]:
        """Calculate upperbound by relaxing integrality constraint."""
        solution_weight = solution.weight
        # If solution weight exceeds capacity, solution is not feasible and has upperbound 0
        if solution_weight > self.capacity:
            return 0
        j = solution.level
        estimate = solution.value
        # Fill knapsack with sorted items until you reach capacity
        while j < self.n and solution_weight + self.weights[j] <= self.capacity:
            estimate += self.values[j]
            solution_weight += self.weights[j]
            j += 1
        # Fill remaining part with fraction left
        if j < self.n:
            estimate += (self.capacity - solution_weight) * (
                self.values[j] / self.weights[j]
            )
        return estimate
