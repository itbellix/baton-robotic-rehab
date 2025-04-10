import heapq
from warnings import warn
import numpy as np


# Credit for Base Astar: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None, strain=None):
        self.parent = parent
        self.position = position
        self.strain = strain

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return "{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f
    

class Astar:

    def __init__(self, N_interp):

        self.N_interp = N_interp        # number of points that the final path should be interpolated to
        self.pe_boundaries = [-20, 160] # defines the interval of physiologically plausible values for the plane of elevation [deg]
        self.se_boundaries = [0, 144]   # as above, for the shoulder elevation [deg]
        self.ar_boundaries = [-90, 100] # as above, for the axial rotation [deg]

        self.strainmap_step = 4         # discretization step used along the model's coordinate [in degrees]
                                        # By default we set it to 4, as the strain maps are generated from the biomechanical model
                                        # with this grid accuracy


    def return_path(self, current_node):
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Return reversed path

    

    def plan(self, maze, start, end, strain, allow_diagonal_movement=True, allow_3D_map=False):
        """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param start:
        :param end:
        :return:
        """
        meanstrain = np.mean(strain)
        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Heapify the open_list and Add the start node
        heapq.heapify(open_list)
        heapq.heappush(open_list, start_node)

        # Adding a stop condition
        outer_iterations = 0
        max_iterations = 2*(len(maze[0]) * len(maze) // 2)

        # what squares do we search
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
        if allow_diagonal_movement:
            adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

        if allow_3D_map:
            max_iterations = (len(maze[0]) * len(maze) * len(maze[2]) // 2)
            adjacent_squares = ((0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                                (-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, 0, -1), (1, 0, 1), (-1, 0, 1), (1, 0, -1),
                                (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1),
                                (1, 0, 0), (-1, 0, 0))

        # Loop until you find the end
        while len(open_list) > 0:
            outer_iterations += 1

            if outer_iterations > max_iterations:
                # if we hit this point return the path such as it is
                # it will not contain the destination
                warn("giving up on pathfinding too many iterations")
                return self.return_path(current_node)

                # Get the current node
            current_node = heapq.heappop(open_list)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                return self.return_path(current_node)

            # Generate children
            children = []

            for new_position in adjacent_squares:  # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                        len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if maze[node_position[0]][node_position[1]] != 0:
                    continue

                # Get Node Strain
                node_strain = strain[node_position[0]][node_position[1]]

                # Create new node
                new_node = Node(current_node, node_position, node_strain)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 20*child.strain #180*meanstrain*child.strain # current_node.g + 1
                # child.g = current_node.g + child.strain #180*meanstrain*child.strain # current_node.g + 1
                child.h = (meanstrain* ((child.position[0] - end_node.position[0])) ** 2) + (meanstrain*((child.position[1] - end_node.position[1])) ** 2)
                child.f = child.g + child.h * (child.strain**2)

                # Child is already in the open list
                if len([open_node for open_node in open_list if
                        child.position == open_node.position and child.g > open_node.g]) > 0:
                    continue

                # Add the child to the open list
                heapq.heappush(open_list, child)
        
        # if no path is found, return None
        return None