import numpy as np
import random

# game config
BOARD_LENGTH = 10
HOLE_POSITION = 0
APPLE_POSITION = 9
START_POSITION = 2
RESTART_POSITION = 3
WIN_REWARD = 100
LOSE_PENALTY = -100
STEP_PENALTY = -1
WIN_POINT = 500
LOSE_POINT = -200

# action
ACTIONS = ['left', 'right']
ACTION_MAP = {'left': -1, 'right': 1}


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((BOARD_LENGTH, len(ACTIONS)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.position = START_POSITION
        self.total_points = 0
        self.reset()

    def reset(self):
        self.position = START_POSITION
        self.total_points = 0

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(ACTIONS))) 
        else:
            return np.argmax(self.q_table[self.position]) 

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def take_action(self, action):
        self.position += ACTION_MAP[ACTIONS[action]]
        if self.position == HOLE_POSITION:
            self.total_points += LOSE_PENALTY
            self.position = RESTART_POSITION
        elif self.position == APPLE_POSITION:
            self.total_points += WIN_REWARD
            self.position = RESTART_POSITION
        else:
            self.total_points += STEP_PENALTY

    def train(self, episodes):
        for episode in range(episodes):
            self.reset()
            while self.total_points < WIN_POINT and self.total_points > LOSE_POINT:
                state = self.position
                action = self.choose_action()
                self.take_action(action)
                next_state = self.position
                reward = self.total_points - STEP_PENALTY if state != HOLE_POSITION and state != APPLE_POSITION else self.total_points
                self.update_q_value(state, action, reward, next_state)

    def show_path(self):
        self.reset()
        path = [self.position]
        while self.total_points < WIN_POINT and self.total_points > LOSE_POINT:
            action = np.argmax(self.q_table[self.position])
            self.take_action(action)
            path.append(self.position)
        print(f"Path: {path}")
        print(f"Total Points: {self.total_points}")
        print("Q-Table:")
        print(self.q_table)


class SARSAAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((BOARD_LENGTH, len(ACTIONS)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.position = START_POSITION
        self.total_points = 0
        self.reset()

    def reset(self):
        self.position = START_POSITION
        self.total_points = 0

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(ACTIONS))) 
        else:
            return np.argmax(self.q_table[self.position]) 

    def update_q_value(self, state, action, reward, next_state, next_action):
        target = reward + self.discount_factor * self.q_table[next_state][next_action]
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

    def take_action(self, action):
        self.position += ACTION_MAP[ACTIONS[action]]
        if self.position == HOLE_POSITION:
            self.total_points += LOSE_PENALTY
            self.position = RESTART_POSITION
        elif self.position == APPLE_POSITION:
            self.total_points += WIN_REWARD
            self.position = RESTART_POSITION
        else:
            self.total_points += STEP_PENALTY

    def train(self, episodes):
        for episode in range(episodes):
            self.reset()
            state = self.position
            action = self.choose_action()
            while self.total_points < WIN_POINT and self.total_points > LOSE_POINT:
                self.take_action(action)
                next_state = self.position
                reward = self.total_points - STEP_PENALTY if state != HOLE_POSITION and state != APPLE_POSITION else self.total_points
                next_action = self.choose_action()
                self.update_q_value(state, action, reward, next_state, next_action)
                state, action = next_state, next_action

    def show_path(self):
        self.reset()
        path = [self.position]
        while self.total_points < WIN_POINT and self.total_points > LOSE_POINT:
            action = np.argmax(self.q_table[self.position])
            self.take_action(action)
            path.append(self.position)
        print(f"Path: {path}")
        print(f"Total Points: {self.total_points}")
        print("Q-Table:")
        print(self.q_table)

if __name__=="__main__":
    # q-learning
    print('Q-LEARNING')
    agent = QLearningAgent()
    agent.train(episodes=1000)
    agent.show_path()
    
    print('\n\n')

    # sarsa
    print('SARSA')
    agent = SARSAAgent()
    agent.train(episodes=1000)
    agent.show_path()

