import heapq
import copy

""" Original author: Jinyue Han
    Source: https://www.youtube.com/watch?v=GuCzYxHa7iA
    The following class was inspired by this video
"""
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        if self.parent is None:
            self.g = 0
            self.parent_index = -1
        else:
            self.g = parent.g + 1
            self.parent_index = self.parent.parent_index + 1
        self.h = cal_h(state)
        self.f = self.g + self.h
        self.children = get_succ(state)


def cal_h(state):
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    h = 0

    if state == goal:
        return h

    for x in range (0, 9):
        if state[x] != goal[x] and state[x] != 0:
            point_wrong = coords[x]
            point_correct = coords[state[x] - 1]
            h += manhattan(point_correct, point_wrong)

    return h


def manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def get_succ(state):
    succ_state = copy.deepcopy(state)
    succ_state_ret = []
    coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    empty_coords = ()

    # finds empty spot
    for x in range(0, 9):
        if state[x] == 0:
            empty_coords = coords[x]

    # 2 succs
    if empty_coords == (0, 0) or empty_coords == (0, 2) or empty_coords == (2, 0) or empty_coords == (2, 2):
        # if top left
        if empty_coords == (0, 0):
            succ_state[0] = state[1]
            succ_state[1] = state[0]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[0] = state[3]
            succ_state[3] = state[0]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

        # if top right
        elif empty_coords == (0, 2):
            succ_state[2] = state[1]
            succ_state[1] = state[2]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[2] = state[5]
            succ_state[5] = state[2]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

        # if bottom left
        elif empty_coords == (2, 0):
            succ_state[6] = state[3]
            succ_state[3] = state[6]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[6] = state[7]
            succ_state[7] = state[6]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

        # if bottom right
        elif empty_coords == (2, 2):
            succ_state[8] = state[5]
            succ_state[5] = state[8]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[8] = state[7]
            succ_state[7] = state[8]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

    # 3 succs
    elif empty_coords == (0, 1) or empty_coords == (1, 0) or empty_coords == (1, 2) or empty_coords == (2, 1):
        # if up
        if empty_coords == (0, 1):
            succ_state[0] = state[1]
            succ_state[1] = state[0]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[1] = state[2]
            succ_state[2] = state[1]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[1] = state[4]
            succ_state[4] = state[1]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

        # if down
        elif empty_coords == (2, 1):
            succ_state[4] = state[7]
            succ_state[7] = state[4]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[6] = state[7]
            succ_state[7] = state[6]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[7] = state[8]
            succ_state[8] = state[7]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

        # if left
        elif empty_coords == (1, 0):
            succ_state[0] = state[3]
            succ_state[3] = state[0]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[3] = state[4]
            succ_state[4] = state[3]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[3] = state[6]
            succ_state[6] = state[3]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

        # if right
        elif empty_coords == (1, 2):
            succ_state[2] = state[5]
            succ_state[5] = state[2]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[4] = state[5]
            succ_state[5] = state[4]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

            succ_state[5] = state[8]
            succ_state[8] = state[5]
            succ_state_ret.append(succ_state)
            succ_state = copy.deepcopy(state)

    # 4 succs
    else:
        succ_state[1] = state[4]
        succ_state[4] = state[1]
        succ_state_ret.append(succ_state)
        succ_state = copy.deepcopy(state)

        succ_state[3] = state[4]
        succ_state[4] = state[3]
        succ_state_ret.append(succ_state)
        succ_state = copy.deepcopy(state)

        succ_state[4] = state[5]
        succ_state[5] = state[4]
        succ_state_ret.append(succ_state)
        succ_state = copy.deepcopy(state)

        succ_state[4] = state[7]
        succ_state[7] = state[4]
        succ_state_ret.append(succ_state)
        succ_state = copy.deepcopy(state)

    return succ_state_ret


def print_succ(state):
    succ_state = get_succ(state)

    for succ in succ_state:
        print (str(succ) + " h=" + str(cal_h(succ)))


def solve(state):
    open = []
    closed = []
    root = Node(state)
    heapq.heappush(open, (root.f, root.state, (root.g, root.h, root)))

    while len(open) != 0:
        n = heapq.heappop(open)
        closed.append(n)

        if n[1] == [1, 2, 3, 4, 5, 6, 7, 8, 0]:
            curr_node = n[2][2]
            node_ret = [curr_node]
            while curr_node.parent is not None:
                node_ret.append(curr_node.parent)
                curr_node = curr_node.parent

            node_ret.reverse()

            for node in node_ret:
                print (str(node.state) + " h=" + str(node.h) + " moves: " + str(node.g))

            # print ("Max queue length: " + str(len(open)))
            return

        else:
            for succ in n[2][2].children:
                open_states = []
                closed_states = []

                for x in open:
                    open_states.append(x[1])
                for x in closed:
                    closed_states.append(x[1])

                # if not in open or closed
                if succ not in open_states and succ not in closed_states:
                    temp_node = Node(succ, n[2][2])
                    heapq.heappush(open, (temp_node.f, temp_node.state, (temp_node.g, temp_node.h, temp_node)))
                # if in open or closed
                else:
                    new_node = Node(succ, n[2][2])
                    if succ in open_states:
                        for x in range (0, len(open_states)):
                            if succ == open_states[x]:
                                old_node = open[x][2][2]
                    elif succ in closed_states:
                        for x in range(0, len(closed_states)):
                            if succ == closed_states[x]:
                                old_node = closed[x][2][2]

                    # comparing g(succ)
                    # if g(succ) is lower for the new version
                    if new_node.g < old_node.g:
                        heapq.heappush(open, (new_node.f, new_node.state, (new_node.g, new_node.h, new_node)))

    return

if __name__ == '__main__':
    solve([4,3,8,5,1,6,7,2,0])
