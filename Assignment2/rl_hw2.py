import random

# 0->south 1->east 2->north 3->west
X = [1, 0, -1, 0]
Y = [0, 1, 0, -1]

iteration = 35000

def firstVisit():
    state = [[0 for i in range(6)] for j in range(6)]
    sum = [[0 for i in range(6)] for j in range(6)]
    count = [[0 for i in range(6)] for j in range(6)]
    # 产生随机序列，起点随机，其中每个点都改
    for times in range(iteration):
        # 确定起点
        cur = random.randint(0, 35)
        if cur == 1 or cur == 35:
            continue
        firstOccur = [-1 for i in range(36)]
        firstOccur[cur] = 0
        episode_len = 0
        for i in range(100):
            x = cur // 6
            y = cur % 6
            # 更新下一状态
            while True:
                action = random.randint(0, 3)
                new_x = x + X[action]
                new_y = y + Y[action]
                if new_x >= 0 and new_x < 6 and new_y >= 0 and new_y < 6:
                    break
            nextstate = 6 * new_x + new_y
            # 记录节点出现位置
            episode_len += 1
            if firstOccur[nextstate] == -1:
                firstOccur[nextstate] = episode_len
            cur = nextstate
            if nextstate == 1 or nextstate == 35:
                break
        # 更新sum和count信息
        if episode_len != 100:
            for i in range(36):
                if firstOccur[i] == -1:
                    continue
                x = i // 6
                y = i % 6
                G = episode_len - firstOccur[i]
                sum[x][y] -= G
                count[x][y] += 1

    print("first-visit monte-carlo:")
    for i in range(6):
        for j in range(6):
            if count[i][j] != 0:
                state[i][j] = sum[i][j] / count[i][j]
        print(state[i])


def everyVisit():
    state = [[0 for i in range(6)] for j in range(6)]
    sum = [[0 for i in range(6)] for j in range(6)]
    count = [[0 for i in range(6)] for j in range(6)]
    # 产生随机序列，起点随机，其中每个点都改
    for times in range(iteration):
        # 确定起点
        cur = random.randint(0, 35)
        if cur == 1 or cur == 35:
            continue
        episode_len = 0
        episode = []
        episode.append(cur)
        for i in range(100):
            x = cur // 6
            y = cur % 6
            # 更新下一状态
            while True:
                action = random.randint(0, 3)
                new_x = x + X[action]
                new_y = y + Y[action]
                if new_x >= 0 and new_x < 6 and new_y >= 0 and new_y < 6:
                    break
            nextstate = 6 * new_x + new_y
            # 记录节点出现位置
            episode_len += 1
            episode.append(nextstate)
            cur = nextstate
            if nextstate == 1 or nextstate == 35:
                break
        # 更新sum和count信息
        if episode_len != 100:
            for i in range(episode_len+1):
                G = episode_len - i
                pos = episode[i]
                x = pos // 6
                y = pos % 6
                sum[x][y] -= G
                count[x][y] += 1

    print("every-visit monte-carlo:")
    for i in range(6):
        for j in range(6):
            if count[i][j] != 0:
                state[i][j] = sum[i][j] / count[i][j]
        print(state[i])


def TD():
    state = [[0 for i in range(6)] for j in range(6)]
    alpha = 0.5
    # 产生随机序列，起点随机，其中每个点都改
    for times in range(iteration):
        # 确定起点
        cur = random.randint(0, 35)
        if cur == 1 or cur == 35:
            continue
        episode_len = 0
        episode = []
        episode.append(cur)
        for i in range(100):
            x = cur // 6
            y = cur % 6
            # 更新下一状态
            while True:
                action = random.randint(0, 3)
                new_x = x + X[action]
                new_y = y + Y[action]
                if new_x >= 0 and new_x < 6 and new_y >= 0 and new_y < 6:
                    break
            nextstate = 6 * new_x + new_y
            # 记录节点出现位置
            episode_len += 1
            episode.append(nextstate)
            cur = nextstate
            if nextstate == 1 or nextstate == 35:
                break
        # 更新sum和count信息
        if episode_len != 100:
            for i in range(episode_len):
                prev = episode[i]
                next = episode[i+1]
                p_x = prev // 6
                p_y = prev % 6
                n_x = next // 6
                n_y = next % 6
                state[p_x][p_y] += alpha * (-1 + state[n_x][n_y] - state[p_x][p_y])

    print("TD:")
    for i in range(6):
        print(state[i])



firstVisit()
everyVisit()
TD()
