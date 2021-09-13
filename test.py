def getRGBfromI(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red, green, blue

# print([i for i in range(5, -1, -1)])

import numpy as np

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
    print(count_2)
    print(count_0)

    return count_2 == count_0


# maze = np.zeros((100, 100)).astype(int)
maze = np.array([[0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0 ,0],
                 [1, 1, 1, 0, 1, 1],
                 [1, 1, 1, 0, 1 ,1],
                 [1, 1, 1, 1, 1, 1]])
# new_maze = check_completeness(maze)
# print(new_maze)
# print(find_zeros(maze, np.array((0, 0)).astype(int)))
# rows, cols = np.where(maze > 1)
# print(len(rows))

def check_in_box(point, box_tl, box_rd):
    if box_tl[0] <= point[0] <= box_rd[0] and box_tl[1] <= point[1] <= box_rd[1]:
        return True
    return False

def filter_array_by_box(array, box_tl, box_rd):
    return array[np.apply_along_axis(lambda point: check_in_box(point, box_tl, box_rd), axis=1, arr = array)]

a = np.array([[1,1], [2, 1], [3, 2], [4, 3]])
print(filter_array_by_box(a, (0, 0), (0, 0)))


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

# Driver Code
a = [[1, 2, 3, 4, 5, 6],
     [7, 8, 9, 10, 11, 12],
     [13, 14, 15, 16, 17, 18]]

R = 3
C = 6

# Function Call
res = spiralPrint(R, C, a)
print(res)
# This code is contributed by Nikita Tiwari.