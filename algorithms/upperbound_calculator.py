import numpy as np
from branch_and_bound import BranchAndBoundSolver, Solution


class UpperBoundCalculator(BranchAndBoundSolver):
    def _calculate_optimistic_estimate_simple(self, solution: Solution) -> int:
        """Calculate upperbound by relaxing capacity constraint."""
        self.cumsum_value = np.insert(np.cumsum([value for value in self.values]), 0, 0)
        next_item_idx = solution.level
        estimate: int = (
            solution.weight + self.cumsum_value[-1] - self.cumsum_value[next_item_idx]
        )
        return estimate

    def _calculate_optimistic_estimate(self, solution: Solution) -> int:
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
