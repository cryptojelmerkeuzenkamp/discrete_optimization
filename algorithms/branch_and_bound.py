"""Branch and bound implementation.

This module shows an implementation of the branch-and-bound algorithm using the depth-first strategy.
Within this optimization we relax the integrality constraint since this will reach faster convergence
compared to the capacity constraint.

You can do an example run by ::
    $python solver.py ./data/ks_4_0

"""


from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Set, Tuple, Union

Solution = namedtuple(
    "Solution", ["level", "value", "weight", "optimistic_estimate", "products"]
)
Item = namedtuple("Item", ["index", "value", "weight"])


class BranchAndBoundSolver(ABC):
    def __init__(self, items: List[Item], capacity: int):
        # Define inputs
        self.items = items
        self.capacity = capacity

        # Initialise solution objects
        self.items_selected = [0] * len(items)
        self.optimal_value = 0
        self.optimal_solution: Set[int] = set()

        # Define values and weights
        self._sort_items()
        self._define_values_and_weights()

        # Define queue with solutions
        self.queue: List[Solution] = []

    def _sort_items(self) -> None:
        """Sort items based on value to weight ratio."""
        self.sorted_items = sorted(
            self.items, key=lambda x: float(x.value) / float(x.weight), reverse=True
        )
        self.n = len(self.sorted_items)

    def _define_values_and_weights(self) -> None:
        """Extracts values and weights from input."""
        self.values = [item.value for item in self.sorted_items]
        self.weights = [item.weight for item in self.sorted_items]

    @abstractmethod
    def _calculate_optimistic_estimate(self, solution: Solution) -> Union[int, float]:
        pass

    def _set_solution(
        self, level: int, value: int, weight: List[int], products: List[int]
    ) -> Solution:
        """Set solution corresponding to a specific node."""
        solution = Solution(
            level,
            value,
            weight,
            0.0,  # Set default upperbound
            products,
        )
        bound = self._calculate_optimistic_estimate(solution)
        solution = solution._replace(
            optimistic_estimate=bound
        )  # Replace default upperbound
        return solution

    def _get_final_solution(self) -> Tuple[int, List[int]]:
        """Set final solution corresponding to optimal node."""
        for item in self.optimal_solution:
            index = self.sorted_items[item - 1].index
            self.items_selected[index] = 1
        total_value = sum(vi * xi for (vi, xi) in zip(self.values, self.items_selected))
        return total_value, self.items_selected

    def _check_if_room_for_extra_item(
        self, new_item: Item, current_solution: Solution
    ) -> bool:
        """Checks whether there is room to add an extra item to the current solution."""
        room = self.capacity - (current_solution.weight + new_item.weight)
        if room < 0:
            return False
        return True

    def _check_if_solution_is_best(self, solution: Solution) -> None:
        """Compare solution to current best, and"""
        if solution.value > self.optimal_value:
            self.optimal_value = solution.value
            self.optimal_solution = set(solution.products)

    def _explore_left(self, current_solution: Solution, level: int) -> None:
        """Explore left pruned part of current solution."""
        # Define new item
        new_item = self.sorted_items[level - 1]

        # Check whether you can add a new item
        if self._check_if_room_for_extra_item(new_item, current_solution):
            # If so, set new solution with new item
            products = current_solution.products + [level]
            solution = self._set_solution(
                level=level,
                value=current_solution.value + new_item.value,
                weight=current_solution.weight + new_item.weight,
                products=products,
            )
            # Check if new solution is best, if so save the solution
            self._check_if_solution_is_best(solution)

            # Explore solution further if upperbound is larger than current value,
            # potentially its value can exceed the current optimal value
            if solution.optimistic_estimate > self.optimal_value:
                self.queue.append(solution)

    def _explore_right(self, current_solution: Solution, level: int) -> None:
        """Explore right pruned part of current solution."""
        solution = self._set_solution(
            level=level,
            value=current_solution.value,
            weight=current_solution.weight,
            products=current_solution.products,
        )
        # Check if new solution is best, if so save the solution
        self._check_if_solution_is_best(solution)

        # Explore solution further if upperbound is larger than current value,
        # potentially its value can exceed the current optimal value
        if solution.optimistic_estimate > self.optimal_value:
            self.queue.append(solution)

    def _explore_tree(self, solution: Solution) -> None:
        """Explore tree based on previous found values."""
        if solution.level < self.n:  # You can only explore until depth N-1
            if (
                solution.optimistic_estimate > self.optimal_value
            ):  # We can't do better than the upperbound
                level = solution.level + 1
                self._explore_left(solution, level)
                self._explore_right(solution, level)

    def execute(self) -> Tuple[int, List[int]]:
        """Execute branch and bound algorithm."""
        # Define initial solution and save in queue
        upperbound = self._calculate_optimistic_estimate(Solution(0, 0, 0, 0, []))
        init_solution = Solution(0, 0, 0, upperbound, [])
        self.queue.append(init_solution)

        while self.queue:
            solution = (
                self.queue.pop()
            )  # Get the solution that entered the queue first (=depth first strategy)
            self._explore_tree(solution=solution)  # Explore solution

        return self._get_final_solution()
