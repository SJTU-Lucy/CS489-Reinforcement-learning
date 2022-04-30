# 0->south 1->east 2->north 3->west
X = [1, 0, -1, 0]
Y = [0, 1, 0, -1]


# v(s, k+1) = sum(p(dir)*(R+v(s', k))) until the max change is limited
def policy_evaluation(state, policy):
    threshold = 0.05
    iteration = 0
    while True:
        iteration += 1
        pre_state = []
        for i in range(6):
            row = []
            for j in range(6):
                row.append(state[i][j])
            pre_state.append(row)
        max_change = 0.0
        for i in range(6):
            for j in range(6):
                state[i][j] = 0
                for k in range(4):
                    x = i + X[k]
                    y = j + Y[k]
                    if x < 0 or x >= 6 or y < 0 or y >= 6:
                        x = i
                        y = j
                    state[i][j] += policy[i][j][k] * (-1 + pre_state[x][y])
                max_change = max(max_change, abs(state[i][j]-pre_state[i][j]))
        if max_change <= threshold:
            break
    print("policy evaluation:")
    print("iter = ", iteration)
    for i in range(6):
        print(state[i])


# after each evaluation, recompute policy until change is made
def policy_iteration(state, policy):
    iteration = 0
    while True:
        iteration += 1
        pre_policy = []
        pre_state = []
        for i in range(6):
            row = []
            p_row = []
            for j in range(6):
                row.append(state[i][j])
                p_row.append(policy[i][j])
            pre_state.append(row)
            pre_policy.append(p_row)
        changed = False
        # evaluate current policy
        for i in range(6):
            for j in range(6):
                state[i][j] = 0
                for k in range(4):
                    x = i + X[k]
                    y = j + Y[k]
                    if x < 0 or x >= 6 or y < 0 or y >= 6:
                        x = i
                        y = j
                    state[i][j] += policy[i][j][k] * (-1 + pre_state[x][y])
        # policy iteration
        for i in range(0, 6):
            for j in range(0, 6):
                if (i == 0 and j == 1) or (i == 5 and j == 5):
                    continue
                Max = -65532
                dir = []
                # find the next direction
                for k in range(4):
                    x = i + X[k]
                    y = j + Y[k]
                    if x < 0 or x >= 6 or y < 0 or y >= 6:
                        x = i
                        y = j
                    if state[x][y] > Max:
                        dir = [k]
                        Max = state[x][y]
                    elif state[x][y] == Max:
                        dir.append(k)
                policy[i][j] = [0, 0, 0, 0]
                # update policy
                for n in dir:
                    policy[i][j][n] = 1 / len(dir)
                # check if policy is updated
                for k in range(4):
                    if policy[i][j][k] != pre_policy[i][j][k]:
                        changed = True
        if changed == False:
            break
    print("policy iteration")
    print("iter = ", iteration)
    for i in range(6):
        print(state[i])
    for i in range(6):
        print(policy[i])


# v(s, k+1) = max(R+v(s', k)) until the max change is limited
def value_iteration(state):
    threshold = 0.01
    iteration = 0
    while True:
        iteration += 1
        pre_state = []
        for i in range(6):
            row = []
            for j in range(6):
                row.append(state[i][j])
            pre_state.append(row)
        max_change = 0.0
        for i in range(6):
            for j in range(6):
                if (i == 0 and j == 1) or (i == 5 and j == 5):
                    continue
                state[i][j] = -65532
                for k in range(4):
                    x = i + X[k]
                    y = j + Y[k]
                    if x < 0 or x >= 6 or y < 0 or y >= 6:
                        x = i
                        y = j
                    state[i][j] = max(state[i][j], -1 + pre_state[x][y])
                max_change = max(max_change, abs(state[i][j]-pre_state[i][j]))
        if max_change <= threshold:
            break
    print("value iteration")
    print("iter = ", iteration)
    for i in range(6):
        print(state[i])


state = []
for i in range(6):
    row = [0, 0, 0, 0, 0, 0]
    state.append(row)

policy = []
for i in range(6):
    row = []
    for j in range(6):
        row.append([0.25, 0.25, 0.25, 0.25])
    policy.append(row)
policy[0][1] = [0, 0, 0, 0]
policy[5][5] = [0, 0, 0, 0]
# policy_evaluation(state, policy)
# policy_iteration(state, policy)
value_iteration(state)