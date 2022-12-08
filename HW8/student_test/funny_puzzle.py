import math
import heapq


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


def swap(state, i, j):
    newlist = state.copy()
    newlist[i] = state[j]
    newlist[j] = state[i]
    return newlist


def count_dist(state):
    count = 0
    size = math.sqrt(len(state))
    for i in range(len(state)):
        if state[i] != 0:
            row = int((state[i]-1) / size)
            col = (state[i]-1) % size
            dist = abs(int(i / size) - row) + abs(i % size - col)
            # print(dist)
            count += dist
    return int(count)


def print_succ(state):
    for i in range(len(state)):
        if state[i] == 0:
            index = i
            break
    succ = []
    #two moves
    size = int(math.sqrt(len(state)))
    row = int(index / size)
    col = int(index % size)
    if row != 0:
        succ.append(swap(state, index, index-size))
    if row != size - 1:
        succ.append(swap(state, index, index+size))
    if col != size - 1:
        succ.append(swap(state, index, index+1))
    if col != 0:
        succ.append(swap(state, index, index-1))
    # if index == 0:
    #     succ.append(swap(state, index, index + 1))
    #     succ.append(swap(state, index, size))
    # elif index == size - 1:
    #     succ.append(swap(state, index, size - 2))
    #     succ.append(swap(state, index, 2 * size-1))
    # elif index == len(state) - size:
    #     succ.append(swap(state, index, size - 2))
    #     succ.append(swap(state, index, 2 * size - 1))
    succ = sorted(succ)
    for i in range(len(succ)):
        print(succ[i], "h={:d}".format(count_dist(succ[i])))

def get_succ(state):
    for i in range(len(state)):
        if state[i] == 0:
            index = i
            break
    succ = []
    #two moves
    size = int(math.sqrt(len(state)))
    row = int(index / size)
    col = int(index % size)
    if row != 0:
        succ.append(swap(state, index, index-size))
    if row != size - 1:
        succ.append(swap(state, index, index+size))
    if col != size - 1:
        succ.append(swap(state, index, index+1))
    if col != 0:
        succ.append(swap(state, index, index-1))
    succ = sorted(succ)
    return succ


def solve(state):
    pq = []
    path = []
    close = []
    open = []
    g = 0
    h = count_dist(state)
    heapq.heappush(pq, (g + h, state, (g, h, -1)))
    path.append((state, -1))
    index = path.index((state, -1))
    sol = heapq.heappop(pq)
    close.append(state)
    Tree = []
    parent = Node(state, None)
    Tree.append(parent)
    F = g + h
    # print(state, "h={:d}".format(h), "moves: {:d}".format(g))
    while sol[2][1] != 0:
        state = sol[1]
        g = sol[2][0] + 1
        successors = get_succ(state)
        for successor in successors:
            openlist = []
            # for
            if successor not in close:
                if successor not in open:
                    h = count_dist(successor)
                    Tree.append(Node(successor, parent))
                    heapq.heappush(pq, (g+h, successor, (g, h, Node(successor, parent))))
                    open.append(successor)

        sol = heapq.heappop(pq)
        state = sol[1]
        h = count_dist(state)
        close.append(state)
        # print(state, "h={:d}".format(h), "moves: {:d}".format(g))
        path.append((state, index))
        index = path.index((state, index))
        parent = sol[2][2]
        # print(index)
    short_cut = []
    # print(path[len(path)-1][0])
    # print(parent.parent.parent.parent.parent.state)
    while parent is not None:
        short_cut.append(parent.state)
        # print(parent.state)
        parent = parent.parent
    short_cut.reverse()
    for i in range(len(short_cut)):
        print(str(short_cut[i]) + " h=" + str(count_dist(short_cut[i])) + " moves: " + str(i))
    # short_cut.append(path[0][0])
    # short_cut.reverse()
    # for i in range(len(short_cut)):
    #     print(short_cut[i], "h={:d}".format(count_dist(short_cut[i])), "moves: {:d}".format(i))
    print("Max queue length: " + str(len(pq)))


if __name__ == '__main__':
    solve([4,3,8,5,1,6,7,2,0])
    # print_succ([1, 0, 3, 4, 5, 8, 7, 2, 6])
    # pq = []
    # heapq.heappush(pq, (5, [1, 2, 3, 4, 5, 0, 6, 7, 8], (0, 5, -1)))
    # print(pq)