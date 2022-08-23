from typing import List
from collections import namedtuple
import queue

Solution = namedtuple('Solution', ['level', 'value', 'weight', 'optimistic_estimate', 'products'])


class BranchAndBoundSolver:

    def __init__(self, items: List, capacity: int):
        # Define inputs
        self.items = items
        self.capacity = capacity

        # Initialise solution objects
        self.items_selected = [0] * len(items)
        self.optimal_value = 0
        self.optimal_solution = set()

        # Define values and weights
        self._sort_items()
        self._define_values_and_weights()

        # Define queue with solutions
        self.q = queue.Queue()

    def _sort_items(self):
        """ Sort items based on value to weight ratio. """
        self.sorted_items = sorted(self.items, key=lambda x: float(x.value) / float(x.weight), reverse=True)
        self.n = len(self.sorted_items)

    def _define_values_and_weights(self):
        """ Extracts values and weights from input. """
        self.values = [item.value for item in self.sorted_items]
        self.weights = [item.weight for item in self.sorted_items]

    def _calculate_optimistic_estimate(self, solution: Solution):
        """ Calculate upperbound by relaxing integer constraint. """
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
            estimate += (self.capacity - solution_weight) * (self.values[j] / self.weights[j])
        return estimate

    def _set_solution(self, level, value, weight, products):
        """ Set solution corresponding to a specific node. """
        solution = Solution(
            level,
            value,
            weight,
            0.0,  # Set default upperbound
            products,
        )
        bound = self._calculate_optimistic_estimate(solution)
        solution = solution._replace(optimistic_estimate=bound)  # Replace default upperbound
        return solution

    def _get_final_solution(self):
        """ Set final solution corresponding to optimal node. """
        for item in self.optimal_solution:
            index = self.sorted_items[item - 1].index
            self.items_selected[index] = 1
        total_value = sum([vi * xi for (vi, xi) in zip(self.values, self.items_selected)])
        return total_value, self.items_selected

    def _check_if_room_for_extra_item(self, new_item, current_solution):
        """ Checks whether there is room to add an extra item to the current solution. """
        room = self.capacity - (current_solution.weight + new_item.weight)
        if room < 0:
            return False
        else:
            return True

    def _check_if_solution_is_best(self, solution: Solution):
        """ Compare solution to current best, and """
        if solution.value > self.optimal_value:
            self.optimal_value = solution.value
            self.optimal_solution = set(solution.products)

    def _explore_left(self, current_solution: Solution, level: int):
        """ Explore left pruned part of current solution. """
        # Define new item
        new_item = self.sorted_items[level-1]

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
                self.q.put(solution)

    def _explore_right(self, current_solution: Solution, level: int):
        """ Explore right pruned part of current solution. """
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
            self.q.put(solution)

    def _explore_tree(self, solution: Solution):
        """ Explore tree based on previous found values. """
        if solution.level < self.n:  # You can only explore until depth N-1
            if solution.optimistic_estimate > self.optimal_value:  # We can't do better than the upperbound
                level = solution.level + 1
                self._explore_left(solution, level)
                self._explore_right(solution, level)

    def execute(self):
        # Define initial solution and save in queue
        upperbound = self._calculate_optimistic_estimate(Solution(0, 0, 0, 0, []))
        init_solution = Solution(0, 0, 0, upperbound, [])
        self.q.put(init_solution)

        while not self.q.empty():
            solution = self.q.get()  # Get the solution that entered the queue first
            self._explore_tree(solution=solution)  # Explore solution

        return self._get_final_solution()
