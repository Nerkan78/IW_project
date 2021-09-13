import argparse
import cv2
from numpy.fft import fft2, ifft2
import numpy as np
from scipy import spatial
from utils import *
import pickle


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

# POSITION_SHIFT = {0 : [(0, 0)],
#                   1: list(filter(lambda x: max(abs(x[0]), abs(x[1])) == 1, [(i, j) for i in range(-1, 2) for j in range(-1,2)])),
#                   2: list(filter(lambda x: max(abs(x[0]), abs(x[1])) == 2, [(i, j) for i in range(-2, 3) for j in range(-2,3)]))}



class Cell:
    def __init__(self, position, status, dirt):
        self.position = position
        self.status = status
        self.dirt = dirt

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class Robot:
    def __init__(self, position, radiance):
        self.position = position
        self.radiance = radiance

    def move(self, end_position):
        self.position = end_position

    def spread_function(self, position, maze):
        line = np.zeros_like(maze).astype(np.uint8)
        cv2.line(line, (self.position[1], self.position[0]), (position[1], position[0]), 1, 1)
        if (np.multiply(line, maze) == 0).all():
            x_shift = position[0] - self.position[0]
            y_shift = position[1] - self.position[1]
            shift = max(abs(x_shift), abs(y_shift))
            if shift < len(self.radiance):
                return self.radiance[shift]
        return 0


class Maze:
    def __init__(self, length, width, robot,  obstacle_number=0, decrease_coef=1, load_maze=False, maze_path=None):
        assert load_maze==False or maze_path is not None, 'Please provide path to file with the maze.'

        if load_maze == False:
            self.length = length
            self.width = width
            self.dirt_map = np.zeros((width, length))

            print('Start maze construction')
            maze = np.zeros((width, length)).astype(np.uint8)
            self.visual = np.ones((width * 5, length * 5, 3)).astype(np.uint8) * 255
            for i in range(obstacle_number):
                print(i)
                success = False

                while not success:
                    print(success)
                    maze_copy = maze.copy()
                    visual_copy = self.visual.copy()
                    x = np.random.randint(length)
                    y = np.random.randint(width)
                    # l = min(np.random.randint(args.length - x), 5)
                    # w = min(np.random.randint(args.length - y), 5)
                    l = np.random.randint(4)
                    w = np.random.randint(4)
                    cv2.rectangle(maze_copy, (x, y), (x + l, y + w), 1, -1)
                    for i in range(x, x + l + 1):
                        for j in range(y, y + w + 1):
                            cv2.rectangle(visual_copy, (i * 5, j * 5), (i * 5 + 5, j * 5 + 5), (127, 127, 127), -1)
                    success = check_completeness(maze_copy)
                maze = maze_copy.copy()
                self.visual = visual_copy.copy()
            print('End maze construction')

        else:
            with open(maze_path, 'rb') as f:
                maze = pickle.load(f)
            self.length = maze.shape[1]
            self.width = maze.shape[0]
            self.dirt_map = np.zeros((self.width, self.length))
            self.visual = np.ones((self.width * 5, self.length * 5, 3)).astype(np.uint8) * 255
            for i in range(self.length):
                for j in range(self.width):
                    if maze[j][i] == 1:
                        cv2.rectangle(self.visual, (i * 5, j * 5), (i * 5 + 5, j * 5 + 5), (127, 127, 127), -1)
        self.maze = maze
        self.cells = []
        for i in range(width):
            cells = []
            for j in range(length):
                cell = Cell((i, j), maze[i][j], 0)
                cells.append(cell)
                cv2.rectangle(self.visual, (j * 5, i * 5), (j * 5 + 5, i * 5 + 5), (0, 0, 0), 1)
            self.cells.append(cells)

        self.decrease_coef = decrease_coef
        self.frames = []
        self.robot = robot


        if maze_path is not None:
            with open(maze_path, 'wb') as f:
                pickle.dump(maze, f)


    def refresh(self):
        self.robot.position = (0, 0)
        self.frames = []
        self.visual = np.ones((self.width * 5, self.length * 5, 3)).astype(np.uint8) * 255

        for i in range(self.width):
            for j in range(self.length):
                self.cells[i][j].dirt = 0
                color = (255, 255, 255) if self.cells[i][j].status == 0 else (127, 127, 127)
                cv2.rectangle(self.visual, (j * 5, i * 5), (j * 5 + 5, i * 5 + 5), color, -1)
        self.dirt_map = np.zeros((self.width, self.length))

    def clean(self, start_point=(0, 0), number_of_steps=-1):
        if number_of_steps == -1:
            number_of_steps = np.inf

        kernel = np.ones((SPREAD_AREA * 2 + 1 , SPREAD_AREA * 2 + 1))
        end = start_point
        boxes = []
        boxes_coordinates = []
        for row_box in range(self.width // 4):
            box_row = []
            for col_box in range(self.length // 4):
                # col = col_box if row_box % 2 == 0 else self.length // 4 - 1 - col_box
                box_row.append([(row_box * 4 , col_box * 4), (row_box * 4 + 4 , col_box * 4 + 4)])
                boxes.append(self.maze[row_box * 4 : row_box * 4 + 4, col_box * 4 : col_box * 4 + 4])
            boxes_coordinates.append(box_row)
        # print(boxes_coordinates)
        boxes_coordinates = spiralPrint(self.width // 4, self.length // 4, boxes_coordinates)
        success = False
        current_box_number = 0
        # for step in range(number_of_steps):
        #     print(step)
        while not success:
            start = end

            max_dist = -1
            min_dist = np.inf
            min_dirt = np.inf

            rows, cols = np.where((self.dirt_map < 100) & (self.maze == 0))
            if len(rows) == 0:
                success = True
                break
            coordinates = np.zeros((len(rows), 2)).astype(int)
            coordinates[:, 0] = rows
            coordinates[:, 1] = cols

            box_tl, box_rd = boxes_coordinates[current_box_number]
            cur_coords = filter_array_by_box(coordinates, box_tl, box_rd)
            print(current_box_number)
            print(cur_coords)
            while len(cur_coords) == 0:
                current_box_number += 1
                box_tl, box_rd = boxes_coordinates[current_box_number]
                cur_coords = filter_array_by_box(coordinates, box_tl, box_rd)
            distance, index = spatial.KDTree(cur_coords).query(self.robot.position)
            end = cur_coords[index]
            # print(end)
            # print('start a star')
            # print(start, end)
            # print(self.maze[tuple(start)])
            # print(self.maze[tuple(end)])

            path = self.astar(tuple(start), tuple(end))
            # print('finish a star')

            # print(path)
            for point in path:
                # print(f'robot position is {self.robot.position}')
                for i in range(self.width):
                    for j in range(self.length):
                        self.cells[i][j].dirt += self.robot.spread_function((i,j), self.maze)
                        self.dirt_map[i][j] += self.robot.spread_function((i,j), self.maze)
                        if self.cells[i][j].status != 1:
                            cv2.rectangle(self.visual, (j * 5, i * 5), (j * 5 + 5, i * 5 + 5), (255, 255-self.cells[i][j].dirt, 255-self.cells[i][j].dirt), -1)
                        # self.cells[i][j].dirt = max(0, self.cells[i][j].dirt-1)
                        # self.dirt_map[i][j] = max(0, self.dirt_map[i][j] - 1)
                        cv2.rectangle(self.visual, (j * 5, i * 5), (j * 5 + 5, i * 5 + 5), (0, 0, 0), 1)

                self.robot.move(point)
                cv2.rectangle(self.visual, (point[1] * 5, point[0] * 5), (point[1] * 5 + 5, point[0] * 5 + 5),
                              (0, 0, 255), -1)
                self.frames.append(self.visual.copy())

    def astar(self, start, end):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]  # Return reversed path

            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0),
                                 (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] >= self.width or node_position[0] < 0 or node_position[1] >= self.length or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if self.cells[node_position[0]][node_position[1]].status != 0:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                flag = False
                for closed_child in closed_list:
                    if child == closed_child:
                        flag = True
                        continue
                if flag:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                flag = False
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        flag = True
                        continue
                if flag:
                    continue

                # Add the child to the open list
                open_list.append(child)


def main():
    parser = argparse.ArgumentParser(description='A_star algorithm for desinfection robot path finding')
    parser.add_argument('--length', help='Length of the maze.', default=100, type=int)
    parser.add_argument('--width', help='Width of the maze.', default=100, type=int)
    # parser.add_argument('--start_point', help='Start point coordinates.', nargs='+', type=int)
    # parser.add_argument('--end_point', help='End point coordinates.', nargs='+', type=int)
    parser.add_argument('--obstacle_number', help='Number of rectangular obstacles in maze', default=0, type=int)
    parser.add_argument('--step_number', help='Number of clean interations', default=-1, type=int)
    parser.add_argument('--spread_area', help='Desinfection Area Spread', default=5, type=int)
    parser.add_argument('--desinfection_rate', help='Power of robot', default=5, type=int)

    args = parser.parse_args()
    global SPREAD_AREA
    global POSITION_SHIFT
    SPREAD_AREA = args.spread_area
    POSITION_SHIFT = {k: list(
        filter(lambda x: max(abs(x[0]), abs(x[1])) == k, [(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1)]))
                      for k in range(SPREAD_AREA)}

    # maze = [[0 for i in range(args.length)] for j in range(args.width)]

    # start = tuple(args.start_point)
    # end = tuple(args.end_point)
    robot = Robot((0, 0), [i * args.desinfection_rate for i in range(SPREAD_AREA-1, -1, -1)])
    print(robot.radiance)
    print(POSITION_SHIFT)
    maze = Maze(args.length, args.width, robot, args.obstacle_number, load_maze=False, maze_path='maze.pkl')
    cv2.imwrite('maze.jpg', maze.visual)
    maze.clean((0, 0), args.step_number)

    record_video(maze.frames, 'video.avi')


if __name__ == '__main__':
    main()

