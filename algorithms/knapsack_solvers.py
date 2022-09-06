"""Knapsack solvers for the well known Knapsack Problem.

This module shows different implementations of Knapsack solvers.
So far implemented:

    - Brand-and-Bound with capacity constraint relaxation.
    - Brand-and-Bound with integrality constraint relaxation.

You can do an example run by, for example, ::
    $value, taken = BranchBoundIntegralityConstraint(items, capacity).execute()

"""

from typing import List, Union

import numpy as np

from algorithms.branch_and_bound import BranchAndBoundSolver, Item, Solution


class BestFirstSolver(BranchAndBoundSolver):
    @staticmethod
    def sort_helper_best_first(sol: Solution) -> int:
        return sol.optimistic_estimate

    def _sort_solution_queue(self) -> None:
        self.queue.sort(key=self.sort_helper_best_first)


class DepthFirstSolver(BranchAndBoundSolver):
    @staticmethod
    def sort_helper_depth_first(sol: Solution) -> int:
        return sol.level

    def _sort_solution_queue(self) -> None:
        self.queue.sort(key=self.sort_helper_depth_first)


class BranchBoundCapacityConstraint(BranchAndBoundSolver):
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


class BranchBoundIntegralityConstraint(BranchAndBoundSolver):
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


# Define all strategy classes
class BranchBoundCapacityConstraintDepthFirst(
    BranchBoundCapacityConstraint, DepthFirstSolver
):
    pass


class BranchBoundCapacityConstraintBestFirst(
    BranchBoundCapacityConstraint, BestFirstSolver
):
    pass


class BranchBoundIntegralityConstraintBestFirst(
    BranchBoundIntegralityConstraint, BestFirstSolver
):
    pass


class BranchBoundIntegralityConstraintDepthFirst(
    BranchBoundIntegralityConstraint, DepthFirstSolver
):
    pass
