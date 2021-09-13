import numpy as np
import cv2


def record_video(array, path, fps=15):
    height, width = array[0].shape[0], array[0].shape[1]
    size = (width, height)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(array)):
        out.write(array[i])
    out.release()


def check_in_box(point, box_tl, box_rd):
    if box_tl[0] <= point[0] < box_rd[0] and box_tl[1] <= point[1] < box_rd[1]:
        return True
    return False


def filter_array_by_box(array, box_tl, box_rd):
    return array[np.apply_along_axis(lambda point: check_in_box(point, box_tl, box_rd), axis=1, arr = array)]


def check_position(maze ,position):
    if maze.shape[0] > position[0] >= 0 and maze.shape[1] > position[1] >= 0 and maze[tuple(position)] == 2:
        return True
    return False


def find_neighbors(maze, position):
    for shift in ([0, 1], [0, -1], [1, 0], [-1, 0]):
        if check_position(maze, position + shift):
            return True
    return False


def check_completeness(maze):
    maze_copy = maze.copy()
    if maze_copy[0][0] != 0:
        return False
    maze_copy[0][0] = 2
    for i in range(maze_copy.shape[0]):
        for j in range(maze_copy.shape[1]):
            if maze_copy[i][j] == 0:
                # if i == maze_copy.shape[0] - 1 or i == 0 or j == maze_copy.shape[1] - 1 or j == 0 or
                if find_neighbors(maze_copy, np.array((i, j))):
                    maze_copy[i][j] = 2
    count_2 = np.count_nonzero(maze_copy == 2)
    count_0 = np.count_nonzero(maze == 0)
    return count_2 == count_0

def spiralPrint(m, n, a):
    k = 0
    l = 0
    result = []

    while (k < m and l < n):
        for i in range(l, n):
            result.append(a[k][i])
        k += 1
        for i in range(k, m):
            result.append(a[i][n - 1])

        n -= 1
        if (k < m):
            for i in range(n - 1, (l - 1), -1):
                result.append(a[m - 1][i])
            m -= 1

        if (l < n):
            for i in range(m - 1, k - 1, -1):
                result.append(a[i][l])
            l += 1
    return result