import random

# 0->south 1->east 2->north 3->west
X = [1, 0, -1, 0]
Y = [0, 1, 0, -1]

iteration = 1000

# ε-greedy pick
def greedypick(Q):
    if Q[0] == 0 and Q[1] == 0 and Q[2] == 0 and Q[3] == 0:
        return random.randint(0, 3)
    # find the largest value
    MaxIndex = -1
    Max = -10000
    for i in range(4):
        if Max < Q[i]:
            Max = Q[i]
            MaxIndex = i
    # choose one
    part = 0.5
    probability = []
    cur = 0
    for i in range(4):
        if i == MaxIndex:
            probability.append(cur + (part / 4 + 1 - part))
            cur += part / 4 + 1 - part
        else:
            probability.append(cur + part / 4)
            cur += part / 4
    rand = random.random()
    if rand <= probability[0]:
        return 0
    if rand <= probability[1]:
        return 1
    if rand <= probability[2]:
        return 2
    else:
        return 3

# cliff: 37~46
# Q(S,A)=Q(S,A)+a(R+Q(S',A')-Q(S,A))
def sarsa():
    Q = [[[0 for i in range(4)] for j in range(12)] for k in range(4)]
    alpha = 0.2
    for times in range(iteration):
        # 从起点开始产生序列
        state = 36
        # 随机产生第一步动作
        action = random.randint(0, 3)
        while state != 47:
            x = state // 12
            y = state % 12
            new_x = x + X[action]
            new_y = y + Y[action]
            if new_x < 0:
                new_x = 0
            if new_x >= 4:
                new_x = 3
            if new_y < 0:
                new_y = 0
            if new_y >= 12:
                new_y = 11
            new_state = new_x * 12 + new_y
            R = -1
            # 如果进入cliff区域，自动回到原点，并且产生-100的反馈值
            if new_state >= 37 and new_state <= 46:
                new_state = 36
                new_x = 3
                new_y = 0
                R = -100
            if new_state == 47:
                R = 100
            # 贪心算法挑选下一步动作
            new_action = greedypick(Q[new_x][new_y])
            tmp_x = new_x + X[new_action]
            tmp_y = new_y + Y[new_action]
            if tmp_x < 0:
                tmp_x = 0
            if tmp_x >= 4:
                tmp_x = 3
            if tmp_y < 0:
                tmp_y = 0
            if tmp_y >= 12:
                tmp_y = 11
            Q[x][y][action] = Q[x][y][action] + alpha * (R + Q[new_x][new_y][new_action] - Q[x][y][action])
            state = new_state
            action = new_action
    print("Sarsa algorithm:")
    cur_state = 36
    result = [["0" for i in range(12)] for j in range(4)]
    while cur_state != 47:
        Action = -1
        Max = -10000
        for i in range(4):
            if Q[cur_state // 12][cur_state % 12][i] == 0:
                continue
            else:
                if Q[cur_state // 12][cur_state % 12][i] > Max:
                    Max = Q[cur_state // 12][cur_state % 12][i]
                    Action = i
        if Action == 0:
            result[cur_state // 12][cur_state % 12] = "↓"
        elif Action == 1:
            result[cur_state // 12][cur_state % 12] = "→"
        elif Action == 2:
            result[cur_state // 12][cur_state % 12] = "↑"
        else:
            result[cur_state // 12][cur_state % 12] = "←"
        x = cur_state // 12 + X[Action]
        y = cur_state % 12 + Y[Action]
        cur_state = 12 * x + y
    for i in range(4):
        print(result[i])


# Q(S,A)=Q(S,A)+a(R+Q(S',Amax)-Q(S,A))
def qlearning():
    Q = [[[0 for i in range(4)] for j in range(12)] for k in range(4)]
    alpha = 0.5
    for times in range(iteration):
        state = 36
        while state != 47:
            x = state // 12
            y = state % 12
            action = greedypick(Q[x][y])
            new_x = x + X[action]
            new_y = y + Y[action]
            if new_x < 0:
                new_x = 0
            if new_x >= 4:
                new_x = 3
            if new_y < 0:
                new_y = 0
            if new_y >= 12:
                new_y = 11
            new_state = new_x * 12 + new_y
            R = -1
            # 如果进入cliff区域，自动回到原点，并且产生-100的反馈值
            if new_state >= 37 and new_state <= 46:
                new_state = 36
                new_x = 3
                new_y = 0
                R = -100
            if new_state == 47:
                R = 100
            max_action = -1
            max_feedback = -10000
            for i in range(4):
                if Q[new_x][new_y][i] > max_feedback:
                    max_action = i
                    max_feedback = Q[new_x][new_y][i]
            Q[x][y][action] = Q[x][y][action] + alpha * (R + Q[new_x][new_y][max_action] - Q[x][y][action])
            state = new_state

    print("Q learning algorithm:")
    cur_state = 36
    result = [["0" for i in range(12)] for j in range(4)]
    while cur_state != 47:
        Action = -1
        Max = -1000
        for i in range(4):
            if Q[cur_state//12][cur_state % 12][i] == 0:
                continue
            else:
                if Q[cur_state//12][cur_state%12][i] > Max:
                    Max = Q[cur_state//12][cur_state%12][i]
                    Action = i
        if Action == 0:
            result[cur_state // 12][cur_state % 12] = "↓"
        elif Action == 1:
            result[cur_state // 12][cur_state % 12] = "→"
        elif Action == 2:
            result[cur_state // 12][cur_state % 12] = "↑"
        else:
            result[cur_state // 12][cur_state % 12] = "←"
        x = cur_state // 12 + X[Action]
        y = cur_state % 12 + Y[Action]
        cur_state = 12 * x + y
    for i in range(4):
        print(result[i])


sarsa()
qlearning()