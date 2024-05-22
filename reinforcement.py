import random
import copy
import numpy as np
import sys
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.patches import Patch

# Configure Python to raise an exception for RuntimeWarning
# warnings.filterwarnings("error", category=RuntimeWarning)


class Sender:
    """
    A Q-learning agent that sends messages to a Receiver

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = range(num_sym)
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = np.zeros(shape=(grid_rows, grid_cols, len(self.actions)))

    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state

        :param state: the state the agent is acting from, in the form (x,y), which are the coordinates of the prize
        :type state: (int, int)
        :return: The symbol to be transmitted (must be an int < N)
        :rtype: int
        """
        explore = np.random.binomial(1, self.epsilon)

        if explore:
            action = np.random.randint(len(self.actions))
        else:
            action = np.argmax(self.q_vals[*state, :])
        return action


    def update_q(self, old_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted, in the form (x,y), which are the coordinates
                          of the prize
        :type old_state: (int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        # v_s_opt = np.max(self.q_vals[*old_state, action], axis=-1)
        self.q_vals[old_state[0], old_state[1], action] += (
            self.alpha * (reward - self.q_vals[old_state[0], old_state[1], action]))

    def update_alpha(self):
        # Update learning rate
        self.alpha -= (self.alpha_i - self.alpha_f) / self.num_ep

class Receiver:
    """
    A Q-learning agent that receives a message from a Sender and then navigates a grid

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = [0,1,2,3] # Note: these correspond to [up, down, left, right]
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = np.zeros(shape=(num_sym, grid_rows, grid_cols, len(self.actions)))

    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state
        :param state: the state the agent is acting from, in the form (m,x,y), where m is the message received
                      and (x,y) are the board coordinates
        :type state: (int, int, int)
        :return: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
        :rtype: int
        """
        explore = np.random.binomial(1, self.epsilon)

        if explore:
            action = random.choice(self.actions)
        else:
            action = np.argmax(self.q_vals[*state, :])
        return action

    def update_q(self, old_state, new_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted in the form (m,x,y), where m is the message received
                          and (x,y) are the board coordinates
        :type old_state: (int, int, int)
        :param new_state: the state the agent entered after it acted
        :type new_state: (int, int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        v_s_opt = np.max(self.q_vals[new_state[0], new_state[1], new_state[2], :])
        self.q_vals[old_state[0], old_state[1], old_state[2], action] += (
            self.alpha * (reward + self.discount * v_s_opt - 
                          self.q_vals[old_state[0], old_state[1], old_state[2], action]))

    def update_alpha(self):
        """
        Updates alpha value
        """
        # Update learning rate
        self.alpha -= (self.alpha_i - self.alpha_f) / self.num_ep

def get_grid(grid_name:str):
    """
    This function produces one of the three grids defined in the assignment as a nested list

    :param grid_name: the name of the grid. Should be one of 'fourroom', 'maze', or 'empty'
    :type grid_name: str
    :return: The corresponding grid, where True indicates a wall and False a space
    :rtype: list[list[bool]]
    """
    grid = [[False for i in range(5)] for j in range(5)] # default case is 'empty'
    if grid_name == 'fourroom':
        grid[0][2] = True
        grid[2][0] = True
        grid[2][1] = True
        grid[2][3] = True
        grid[2][4] = True
        grid[4][2] = True
    elif grid_name == 'maze':
        grid[1][1] = True
        grid[1][2] = True
        grid[1][3] = True
        grid[2][3] = True
        grid[3][1] = True
        grid[4][1] = True
        grid[4][2] = True
        grid[4][3] = True
        grid[4][4] = True
    return grid


def legal_move(posn_x:int, posn_y:int, move_id:int, grid:list[list[bool]]):
    """
    Produces the new position after a move starting from (posn_x,posn_y) if it is legal on the given grid (i.e. not
    out of bounds or into a wall)

    :param posn_x: The x position (column) from which the move originates
    :type posn_x: int
    :param posn_y: The y position (row) from which the move originates
    :type posn_y: int
    :param move_id: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
    :type move_id: int
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :return: The new (x,y) position if the move was legal, or the old position if it was not
    :rtype: (int, int)
    """
    moves = [[0,-1],[0,1],[-1,0],[1,0]]
    new_x = posn_x + moves[move_id][0]
    new_y = posn_y + moves[move_id][1]
    result = (new_x,new_y)
    if new_x < 0 or new_y < 0 or new_x >= len(grid[0]) or new_y >= len(grid):
        result = (posn_x,posn_y)
    else:
        if grid[new_y][new_x]:
            result = (posn_x,posn_y)
    return result


def run_episodes(sender:Sender, receiver:Receiver, grid:list[list[bool]], num_ep:int, delta:float, ep_frame_freq:int=0):
    
    """
    Runs the reinforcement learning scenario for the specified number of episodes

    :param sender: The Sender agent
    :type sender: Sender
    :param receiver: The Receiver agent
    :type receiver: Receiver
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :param num_ep: The number of episodes
    :type num_ep: int
    :param delta: The chance of termination after every step of the receiver
    :type delta: float [0,1]
    :return: A list of the reward received by each agent at the end of every episode
    :rtype: list[float]
    """
    reward_vals = []
    paths = []
    signals = []
    prize_positions = []
    frames = []

    progress_bar = tqdm(total=num_ep, desc='Training')

    # Episode loop
    for ep in range(num_ep):
        # Set receiver starting position
        receiver_x = 2
        receiver_y = 2

        # Choose prize position
        prize_x = np.random.randint(len(grid[0]))
        prize_y = np.random.randint(len(grid))
        while grid[prize_y][prize_x] or (prize_x == receiver_x and prize_y == receiver_y):
            prize_x = np.random.randint(len(grid[0]))
            prize_y = np.random.randint(len(grid))

        # Initialize new episode
        # (sender acts)
        send_state = (prize_x, prize_y)
        sym = sender.select_action(send_state)

        # Receiver loop
        # (receiver acts, check for prize, check for random termination, update receiver Q-value)
        terminate = False
        reward = 0
        discounted_reward = 0
        step = 0
        # print(f"Prize postion: {prize_x, prize_y}")
        episode_frames = []
        if ep_frame_freq > 0:
            ep_frame_counter = ep // ep_frame_freq  # Counter of framed episodes
        while not terminate:
            step += 1
            rec_state = (sym, receiver_x, receiver_y)
            move_id = receiver.select_action(rec_state)
            receiver_x,  receiver_y = legal_move(receiver_x, receiver_y, move_id, grid)

            # Set random termination
            terminate = np.random.binomial(1, delta)

            # Check for the reward
            if (receiver_x,  receiver_y) == (prize_x, prize_y):
                # Applying discount based on the number of steps that has taken the receiver to reach the prize
                reward = 1
                discounted_reward = reward * (receiver.discount ** (step-1))
                terminate = True
            rec_new_state = (sym, receiver_x, receiver_y)
            receiver.update_q(rec_state, rec_new_state, move_id, reward)

            if ep_frame_freq > 0:
                if ep % ep_frame_freq == 0:
                    episode_frames.append((receiver_x, receiver_y, prize_x, prize_y, copy.deepcopy(receiver), copy.deepcopy(sender), sym, reward, ep_frame_counter))
                    
            # print(f"Receiver old state: {rec_state} Receiver new state: {rec_new_state}")

        #Finish up episode
        # (update sender Q-value, update alpha values, append reward to output list)
        sender.update_q(send_state, sym, reward)
        receiver.update_alpha()
        sender.update_alpha()
        reward_vals.append(discounted_reward)

        if ep_frame_freq > 0:
            if ep % ep_frame_freq == 0:
                frames.append(episode_frames)
        
        progress_bar.update(1)

    progress_bar.close()

    return reward_vals, receiver, sender, frames


def plot_sender_policy(sender):
    fourroom = [[2, 0], [2, 4], [0, 2], [1, 2], [3, 2], [4, 2]]
    max_q_values = np.max(sender.q_vals, axis=2) # Max Q-value for any action at each position
    max_q_action = np.argmax(sender.q_vals, axis=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw the grid lines
    for x in range(max_q_values.shape[0] + 1):
        ax.axhline(x, lw=2, color='k')
    for y in range(max_q_values.shape[1] + 1):
        ax.axvline(y, lw=2, color='k')

    # Place the Q-values as text in the center of each cell
    for i in range(max_q_values.shape[0]):
        for j in range(max_q_values.shape[1]):
            if [i, j] in fourroom:
                ax.text(i + 0.5, j + 0.5, "\u25A0",
                        horizontalalignment='center',
                        verticalalignment='center')
            else:
                ax.text(i + 0.5, j + 0.5, '{:d}'.format(max_q_action[i, j]),
                    horizontalalignment='center',
                    verticalalignment='center')

    plt.title('Sender Policy')
    plt.gca().invert_yaxis()
    plt.show()

def plot_receiver_policy(receiver, signal_num):
    # Plot a separate grid for each signal
    fourroom = [[2, 0], [2, 4], [0, 2], [1, 2], [3, 2], [4, 2]]
    map_arrow = {0: '\u2191', 1:'\u2193', 2:'\u2190', 3:'\u2192'}
    for signal in range(signal_num):
        max_q_values = np.max(receiver.q_vals[signal], axis=2)  # Max Q-value for each action at each position for the given signal
        max_q_action = np.argmax(receiver.q_vals[signal], axis=2)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.axis('off')

        # Drawing the grid lines
        for x in range(max_q_values.shape[0] + 1):
            ax.axhline(x, lw=2, color='k')
        for y in range(max_q_values.shape[1] + 1):
            ax.axvline(y, lw=2, color='k')

        # Placing the Q-values as text in the center of each cell
        for i in range(max_q_values.shape[0]):
            for j in range(max_q_values.shape[1]):
                if [i, j] in fourroom:
                    ax.text(i + 0.5, j + 0.5, "\u25A0",
                            horizontalalignment='center',
                            verticalalignment='center')
                else:
                    ax.text(i + 0.5, j + 0.5, map_arrow[max_q_action[i, j]],
                            horizontalalignment='center',
                            verticalalignment='center')

        plt.title(f'Receiver Policy for Signal {signal}')
        plt.gca().invert_yaxis()
        plt.show()

def scenario_e():

    # Define parameters here
    grid_name = 'empty' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    epsilon = 0.1
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01

    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    num_signals = [1]
    n_tests = 10

    # n_eps = [10, 100, 1000]
    # epsilons = [0.01, 0.1]
    # n_tests = 4

    rewards = np.zeros(shape=(len(n_eps), len(num_signals), n_tests))
    log_n_ep = np.array([np.log10(n_ep) for n_ep in n_eps])

    for idx_n_ep, n_ep in enumerate(n_eps):
        for idx_num_signal, num_signal in enumerate(num_signals):
            for test in range(n_tests):
                # print("\033[A\033[K", end="")
                # print(f"n_ep: {n_ep}, num_signal: {num_signal}, test: {test}")
                # Initialize agents
                sender = Sender(num_signal, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)
                receiver = Receiver(num_signal, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)

                # Learn
                learn_rewards, _, _ = run_episodes(sender, receiver, grid, n_ep, delta)


                # Define parameters here
                sender.epsilon = 0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                n_ep_test = 1000

                reward, _, _ = run_episodes(sender, receiver, grid, n_ep_test, delta)
                rewards[idx_n_ep, idx_num_signal, test] = np.mean(reward)

    fig, ax = plt.subplots()
    for idx_num_signal in range(rewards.shape[1]):
        ax.errorbar(log_n_ep, rewards[:, idx_num_signal, :].mean(axis=-1),
                    yerr=rewards[:, idx_num_signal, :].std(axis=-1), capsize=5,
                    label=f'Number of Signals = {num_signals[idx_num_signal]}',
                    alpha=1, fmt='-o', lw=0.5)
        ax.set_title(f"Average Discounted Reward")
        ax.set_xlabel("log(Number of Episodes)")
        ax.set_ylabel("Average Discounted Reward")
        ax.legend(loc='upper left', fontsize='small')
    fig.savefig("avg_discounted_rewards_e.pdf")
    # print("scenario e is completed!")
    plt.show()

def scenario_d():

    # Define parameters here
    grid_name = 'maze' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    epsilon = 0.1
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01

    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    num_signals = [2, 3, 5]
    n_tests = 10

    # n_eps = [10, 100, 1000]
    # epsilons = [0.01, 0.1]
    # n_tests = 4

    rewards = np.zeros(shape=(len(n_eps), len(num_signals), n_tests))
    log_n_ep = np.array([np.log10(n_ep) for n_ep in n_eps])

    for idx_n_ep, n_ep in enumerate(n_eps):
        for idx_num_signal, num_signal in enumerate(num_signals):
            for test in range(n_tests):
                # print("\033[A\033[K", end="")
                # print(f"n_ep: {n_ep}, num_signal: {num_signal}, test: {test}")
                # Initialize agents
                sender = Sender(num_signal, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)
                receiver = Receiver(num_signal, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)

                # Learn
                learn_rewards, _, _ = run_episodes(sender, receiver, grid, n_ep, delta)

                # Define parameters here
                sender.epsilon = 0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                n_ep_test = 1000

                reward, _, _ = run_episodes(sender, receiver, grid, n_ep_test, delta)
                rewards[idx_n_ep, idx_num_signal, test] = np.mean(reward)

    fig, ax = plt.subplots()
    for idx_num_signal in range(rewards.shape[1]):
        ax.errorbar(log_n_ep, rewards[:, idx_num_signal, :].mean(axis=-1),
                    yerr=rewards[:, idx_num_signal, :].std(axis=-1), capsize=3,
                    label=f'Number of Signals = {num_signals[idx_num_signal]}',
                    alpha=1, fmt='-o', lw=0.5)
        ax.set_title(f"Average Discounted Reward")
        ax.set_xlabel("log(Number of Episodes)")
        ax.set_ylabel("Average Discounted Reward")
        ax.legend(loc='upper left', fontsize='small')
    fig.savefig("avg_discounted_rewards_d.pdf")
    # print("scenario d is completed!")
    plt.show()

def scenario_c():

    # Define parameters here
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    epsilon = 0.1
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01

    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    num_signals = [2, 4, 10]
    n_tests = 10

    # n_eps = [10, 100, 1000]
    # epsilons = [0.01, 0.1]
    # n_tests = 4

    rewards = np.zeros(shape=(len(n_eps), len(num_signals), n_tests))
    log_n_ep = np.array([np.log10(n_ep) for n_ep in n_eps])

    for idx_n_ep, n_ep in enumerate(n_eps):
        for idx_num_signal, num_signal in enumerate(num_signals):
            for test in range(n_tests):
                # print("\033[A\033[K", end="")
                # print(f"n_ep: {n_ep}, num_signal: {num_signal}, test: {test}")
                # Initialize agents
                sender = Sender(num_signal, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)
                receiver = Receiver(num_signal, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)

                # Learn
                learn_rewards, _, _ = run_episodes(sender, receiver, grid, n_ep, delta)

                # Define parameters here
                sender.epsilon = 0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                n_ep_test = 1000

                reward, _, _ = run_episodes(sender, receiver, grid, n_ep_test, delta)
                rewards[idx_n_ep, idx_num_signal, test] = np.mean(reward)

    fig, ax = plt.subplots()
    for idx_num_signal in range(rewards.shape[1]):
        ax.errorbar(log_n_ep, rewards[:, idx_num_signal, :].mean(axis=-1),
                    yerr=rewards[:, idx_num_signal, :].std(axis=-1), capsize=3,
                    label=f'Number of Signals = {num_signals[idx_num_signal]}',
                    alpha=1, fmt='-o', lw=0.5)
        ax.set_title(f"Average Discounted Reward")
        ax.set_xlabel("log(Number of Episodes)")
        ax.set_ylabel("Average Discounted Reward")
        ax.legend(loc='upper left', fontsize='small')
    fig.savefig("avg_discounted_rewards_c.pdf")
    # print("scenario c is completed!")
    plt.show()

def scenario_b():

    # Define parameters here
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01

    n_eps = [10, 100, 1000, 10000, 50000, 100000]
    epsilons = [0.01, 0.1, 0.4]
    n_tests = 10

    # n_eps = [100000]
    # epsilons = [0.1]
    # n_tests = 1

    rewards = np.zeros(shape=(len(n_eps), len(epsilons), n_tests))
    log_n_ep = np.array([np.log10(n_ep) for n_ep in n_eps])

    for idx_n_ep, n_ep in enumerate(n_eps):
        for idx_epsilon, epsilon in enumerate(epsilons):
            for test in range(n_tests):
                # print("\033[A\033[K", end="")
                # print(f"n_ep: {n_ep}, epsilon: {epsilon}, test: {test}")
                # Initialize agents
                sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)
                receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, n_ep, epsilon, discount)

                # Learn
                learn_rewards, _, _ = run_episodes(sender, receiver, grid, n_ep, delta)

                # Define parameters here
                sender.epsilon = 0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                n_ep_test = 1000

                reward, receiver, sender = run_episodes(sender, receiver, grid, n_ep_test, delta)
                rewards[idx_n_ep, idx_epsilon, test] = np.mean(reward)

    fig, ax = plt.subplots()
    for idx_epsilon in range(rewards.shape[1]):
        ax.errorbar(log_n_ep, rewards[:, idx_epsilon, :].mean(axis=-1),
                    yerr=rewards[:, idx_epsilon, :].std(axis=-1), capsize=3,
                    label=f'Epsilon = {epsilons[idx_epsilon]}',
                    alpha=0.7, fmt='-o', lw=0.5)
        ax.set_title(f"Average Discounted Reward")
        ax.set_xlabel("log(Number of Episodes)")
        ax.set_ylabel("Average Discounted Reward")
        ax.legend(loc='upper left', fontsize='small')
    fig.savefig("avg_discounted_rewards_b.pdf")
    # print("scenario b is completed!")
    plt.show()

    plot_receiver_policy(receiver, signal_num=4)
    plot_sender_policy(sender)


def default_scenario():
    # Define parameters here
    num_learn_episodes = 100000
    num_test_episodes = 1000
    grid_name = 'fourroom' # 'fourroom', 'maze', or 'empty'
    grid = get_grid(grid_name)
    num_signals = 25
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01

    # Initialize agents
    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

    # Learn
    learn_rewards, _, _ = run_episodes(sender, receiver, grid, num_learn_episodes, delta)

    # Test
    sender.epsilon = 0.0
    sender.alpha = 0.0
    sender.alpha_i = 0.0
    sender.alpha_f = 0.0
    receiver.epsilon = 0.0
    receiver.alpha = 0.0
    receiver.alpha_i = 0.0
    receiver.alpha_f = 0.0
    test_rewards, _, _ = run_episodes(sender, receiver, grid, num_test_episodes, delta)

    # Print results
    print("Average discounted reward during learning: " + str(np.average(learn_rewards)))
    print("Average discounted reward during testing: " + str(np.average(test_rewards)))


def animate_sender_receiver(frames, grid, savepath='learning_stage.mp4', fps=20):
    """
    Creates and stores an animation of training episodes. The sender is represented as a flower that also represents the location of the prize
    and the receivers is shown as a bee that uses the signal to pipoint the sender (prize).

    param frames: A list that represents frames episodes. Each framed episode is itself a list of frames with each frame representing the data
    required to reconstruct the frame
    type frames: List 
    param grid: The grid in which the search takes place
    type grid: List
    param savepath: The path of the animation file
    type savepath: String
    param fps: frame per second of the animation
    type fps: Integer
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [15, 1]})  # Added second subplot for the progress bar
    # Remove ticks and labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax1.set_facecolor('#F4F4F4')  # Light gray background

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0, len(frames))
    ax2.set_ylim(0, 1)
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax2.set_facecolor('#F4F4F4')  # Light gray background

    receiver_img = mpimg.imread('bee.png')
    sender_img = mpimg.imread('flower.png')
    reward_img = mpimg.imread('honey.png')

    fourroom = [[2, 0], [2, 4], [0, 2], [1, 2], [3, 2], [4, 2]]

    symbol_colors = {
        0: '#FF9999',  # Light red
        1: '#99FF99',  # Light green
        2: '#9999FF',  # Light blue
        3: '#FFFF99'   # Light yellow
    }
    arrow_directions = {
        0: '\u2191',  # Up
        1: '\u2193',  # Down
        2: '\u2190',  # Left
        3: '\u2192'   # Right
    }

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=sum([len(episode_frames) for episode_frames in frames]), desc='Generating animation')

    def update(frame):
        # Extract frame data
        receiver_x, receiver_y, prize_x, prize_y, receiver, sender, sym, reward, ep_frame_counter = frame

        ax1.clear()  # Clear only the patches
        while ax1.texts:
            del ax1.texts[0]


        ax1.set_xlim(-0.5, len(grid[0]) - 0.5)
        ax1.set_ylim(-0.5, len(grid) - 0.5)

        # Draw grid walls
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x]:
                    ax1.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=True, color="black", zorder=3))
                else:
                    ax1.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, color=None, zorder=3))


        # Scale the receiver and sender image
        played_frames_ratio = ep_frame_counter / len(frames)
        scale = 0.2 + 0.2 * played_frames_ratio  # Increase the scale factor over episodes

        # Draw the prize, receiver and sender
        if reward==1:
            prize_image = ax1.imshow(reward_img, extent=(prize_x-scale, prize_x+scale, prize_y+scale, prize_y-scale))
            prize_image.set_zorder(6)
        else:
            sender_image = ax1.imshow(sender_img, extent=(prize_x-scale, prize_x+scale, prize_y+scale, prize_y-scale))
            sender_image.set_zorder(4)
        
            receiver_image = ax1.imshow(receiver_img, extent=(receiver_x-scale, receiver_x+scale, receiver_y+scale, receiver_y-scale))
            receiver_image.set_zorder(5)

        # Fill the sender grid with colors
        max_q_action = np.argmax(sender.q_vals, axis=2)
        for i in range(sender.q_vals.shape[0]):
            for j in range(sender.q_vals.shape[1]):
                symbol = max_q_action[i, j]
                color = symbol_colors.get(symbol, 'white')  # Default to white if symbol is not in dictionary
                ax1.add_patch(plt.Rectangle((i - 0.4, j - 0.4), 0.8, 0.8, fill=True, color=color, zorder=1))

        # Add signals to the sender grid
        max_q_action = np.argmax(receiver.q_vals[sym], axis=2)

       # Placing the Q-values as text in the center of each cell
        for i in range(max_q_action.shape[0]):
            for j in range(max_q_action.shape[1]):
                if [i, j] in fourroom:
                    ax1.text(i, j, "\u25A0",
                            ha='center', va='center',
                            fontsize=24, zorder=2)
                else:
                    ax1.text(i, j, arrow_directions[max_q_action[i, j]],
                            ha='center', va='center',
                            fontsize=24, zorder=2)

        # Set title and invert y-axis
        ax1.set_title('Receiver and Sender', fontsize=20, fontweight='bold')
        ax1.invert_yaxis()

        # Add symbol coloring legend
        legend_elements = [Patch(facecolor=symbol_colors.get(key), edgecolor='black', label=f'Symbol {key}')
                           for key in symbol_colors]
        legend = ax1.legend(handles=legend_elements, loc='center left', title='4-symbol Map', frameon=True, facecolor='white', framealpha=1)
        plt.setp(legend.get_texts(), color='black')

        # Add second legend for current symbol
        second_legend_element = [Patch(facecolor=symbol_colors.get(sym), edgecolor='black', label=f'Signal: {sym}')]
        second_legend = ax1.legend(handles=second_legend_element, loc='center right', title='Flower Signal', frameon=True, facecolor='white', framealpha=1)
        ax1.add_artist(legend)  # Ensure the first legend is also displayed
        plt.setp(second_legend.get_texts(), color='black')

        # Draw progress bar
        ax2.add_patch(plt.Rectangle((0, 0), ep_frame_counter, 1, fill=True, color='#0066CC', edgecolor=None))
        ax2.set_title('The Training Progress', fontsize=20, fontweight='bold')
        # ax2.text(ep_frame_counter / 2, 0.5, f'{ep_frame_counter}', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        ax2.set_xlim(0, len(frames))
        ax2.set_ylim(0, 1)

        # Update progress bar
        progress_bar.update(1)

    anim = animation.FuncAnimation(fig, update, frames=[frame for episode_frames in frames for frame in episode_frames], repeat=False)
    anim.save(savepath, writer='ffmpeg', fps=fps)
    progress_bar.close()
    print("RL training animation is saved!")


def run_with_animation():
    grid_name = 'fourroom'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01
    num_ep = 100000
    epsilon = 0.1

    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)

    reward_vals, receiver, sender, frames = run_episodes(sender, receiver, grid, num_ep, delta, ep_frame_freq=500)
    animate_sender_receiver(frames, grid, fps=5)


if __name__ == "__main__":
    # default_scenario()
    run_with_animation()
    # scenario_b()
    # scenario_c()
    # scenario_d()
    # scenario_e()

