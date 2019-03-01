import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

class Bandit(object):
    def __init__(self, arms, std=1.0, stationary=True):
        self.arms = arms
        self.std = std
        self.stationary = stationary
        self.perturbation_sigma = 1e-2

        self.min_value = -2
        self.max_value = 2
        if not self.stationary: # All bandits start equal
            self.bandits = np.zeros(shape=(self.arms))
        else:
            self.bandits = np.random.uniform(self.min_value, self.max_value+1, self.arms)

    def get_optimal_action(self):
        return np.argmax(self.bandits)

    def __str__(self):
        return "{}-armed bandit - Optimal action: {}".format(self.arms, self.get_optimal_action())

    def act(self, action):
        if self.stationary:
            return self.std * np.random.randn() + self.bandits[action]
        else:
            # Add perturbation
            self.bandits = self.bandits + self.perturbation_sigma * np.random.randn(self.arms)
            return self.std * np.random.randn() + self.bandits[action]

def epsilon_greedy_policy(Q_a, epsilon, num_actions):
    random = np.random.rand() < epsilon

    if random:
        return np.random.randint(0, num_actions)
    else:
        # Argmax with random selection when ties
        return np.random.choice(np.flatnonzero(Q_a == Q_a.max()))

def UCB(Q_a, N_a, num_actions, t, c):
    """
        - Q_a: current action value estimate
        - N_a: Visit count per action
        - num_actions:
        - t: current step
        - c: degree of exploration
    """
    
    # See equation 2.10
    new_Q_a = Q_a + c * np.sqrt(np.log(t + 1) / (N_a + 1e-5))
    # Argmax with random selection when ties
    return np.random.choice(np.flatnonzero(new_Q_a == new_Q_a.max()))

def stationary_env():
    """
    Implementation of Figure 2.2 from Chapter 2.4
    """
    arms = 10
    num_steps = 1000
    num_runs = 2000
    
    epsilons = [0, 0.001, 0.01, 0.1]

    rewards = np.zeros(shape=(len(epsilons), num_runs, num_steps))
    optimal_actions = np.zeros(shape=(len(epsilons), num_runs, num_steps))

    for idx, epsilon in enumerate(epsilons):
        for run in tqdm(range(num_runs)):
            bandit = Bandit(arms)
            Q_a = np.zeros(arms)
            N_a = np.zeros(arms)
                
            for step in range(num_steps):
                action = epsilon_greedy_policy(Q_a, epsilon, arms)
                reward = bandit.act(action)
                N_a[action] = N_a[action] + 1
                Q_a[action] = Q_a[action] + (1. / N_a[action]) * (reward - Q_a[action]) 
                rewards[idx][run][step] = reward
                optimal_actions[idx][run][step] = 1 if action == bandit.get_optimal_action() else 0

        plt.subplot(121)
        plt.plot(np.mean(rewards[idx], axis=0), label='epsilon={}'.format(epsilon))
        plt.xlabel('Steps', fontsize=18)
        plt.ylabel('Average reward', fontsize=18)
        plt.legend(loc='best', shadow=True, fancybox=True)

        plt.subplot(122)
        plt.plot(np.mean(optimal_actions[idx], axis=0), label='epsilon={}'.format(epsilon))
        plt.xlabel('Steps', fontsize=18)
        plt.ylabel('% Optimal action', fontsize=18)
        plt.legend(loc='best', shadow=True, fancybox=True)
    
    plt.show()

def non_stationary_env():
    """
    Implementation of exercice 2.5
    """

    arms = 10
    num_steps = 10000
    num_runs = 500
    
    epsilon = 0.1

    # Incremental alpha:
    rewards = np.zeros(shape=(num_runs, num_steps))
    optimal_actions = np.zeros(shape=(num_runs, num_steps))

    for run in tqdm(range(num_runs)):
        bandit = Bandit(arms, stationary=False)
        Q_a = np.zeros(arms)
        N_a = np.zeros(arms)
            
        for step in range(num_steps):
            action = epsilon_greedy_policy(Q_a, epsilon, arms)
            reward = bandit.act(action)
            N_a[action] = N_a[action] + 1
            Q_a[action] = Q_a[action] + (1. / N_a[action]) * (reward - Q_a[action]) 
            rewards[run][step] = reward
            optimal_actions[run][step] = 1 if action == bandit.get_optimal_action() else 0


    plt.subplot(121)
    plt.plot(np.mean(rewards, axis=0), label='Incremental alpha')
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Average reward', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    plt.subplot(122)
    plt.plot(np.mean(optimal_actions, axis=0), label='Incremental alpha')
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('% Optimal action', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    # Fixed alpha:
    alpha = 0.1
    rewards = np.zeros(shape=(num_runs, num_steps))
    optimal_actions = np.zeros(shape=(num_runs, num_steps))

    for run in tqdm(range(num_runs)):
        bandit = Bandit(arms, stationary=False)
        Q_a = np.zeros(arms)
        N_a = np.zeros(arms)
            
        for step in range(num_steps):
            action = epsilon_greedy_policy(Q_a, epsilon, arms)
            reward = bandit.act(action)
            N_a[action] = N_a[action] + 1
            Q_a[action] = Q_a[action] + alpha * (reward - Q_a[action]) 
            rewards[run][step] = reward
            optimal_actions[run][step] = 1 if action == bandit.get_optimal_action() else 0


    plt.subplot(121)
    plt.plot(np.mean(rewards, axis=0), label='Fixed alpha')
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Average reward', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    plt.subplot(122)
    plt.plot(np.mean(optimal_actions, axis=0), label='Fixed alpha')
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('% Optimal action', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    plt.show()

def stationary_env_UCB():
    arms = 10
    num_steps = 1000
    num_runs = 2000
    

    # Standard epsilon greedy
    epsilon = 0.1

    rewards = np.zeros(shape=(num_runs, num_steps))
    optimal_actions = np.zeros(shape=(num_runs, num_steps))

    for run in tqdm(range(num_runs)):
        bandit = Bandit(arms)
        Q_a = np.zeros(arms)
        N_a = np.zeros(arms)
            
        for step in range(num_steps):
            action = epsilon_greedy_policy(Q_a, epsilon, arms)
            reward = bandit.act(action)
            N_a[action] = N_a[action] + 1
            Q_a[action] = Q_a[action] + (1. / N_a[action]) * (reward - Q_a[action]) 
            rewards[run][step] = reward
            optimal_actions[run][step] = 1 if action == bandit.get_optimal_action() else 0
    plt.subplot(121)
    plt.plot(np.mean(rewards, axis=0), label='epsilon={}'.format(epsilon))
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Average reward', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    plt.subplot(122)
    plt.plot(np.mean(optimal_actions, axis=0), label='epsilon={}'.format(epsilon))
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('% Optimal action', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)


    # UCB
    rewards = np.zeros(shape=(num_runs, num_steps))
    optimal_actions = np.zeros(shape=(num_runs, num_steps))

    for run in tqdm(range(num_runs)):
        bandit = Bandit(arms)
        Q_a = np.zeros(arms)
        N_a = np.zeros(arms)
            
        for step in range(num_steps):
            action = UCB(Q_a, N_a, arms, step, 2.0)
            reward = bandit.act(action)
            N_a[action] = N_a[action] + 1
            Q_a[action] = Q_a[action] + (1. / N_a[action]) * (reward - Q_a[action]) 
            rewards[run][step] = reward
            optimal_actions[run][step] = 1 if action == bandit.get_optimal_action() else 0

    plt.subplot(121)
    plt.plot(np.mean(rewards, axis=0), label='UCB with c={}'.format(2.0))
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Average reward', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    plt.subplot(122)
    plt.plot(np.mean(optimal_actions, axis=0), label='UCB with c={}'.format(2.0))
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('% Optimal action', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)


    plt.show()

def gradient_bandit():
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    arms = 10
    num_steps = 1000
    num_runs = 200    
    alpha = 0.1

    rewards = np.zeros(shape=(num_runs, num_steps))
    optimal_actions = np.zeros(shape=(num_runs, num_steps))
    for run in tqdm(range(num_runs)):
        bandit = Bandit(arms)
        H_a = np.zeros(arms)
        pi_a = softmax(H_a)
        Q_a = np.zeros(arms)
        N_a = np.zeros(arms)
            
        for step in range(num_steps):
            action = np.random.choice(range(arms), p=pi_a)

            reward = bandit.act(action)
            N_a[action] = N_a[action] + 1
            Q_a[action] = Q_a[action] + (1. / N_a[action]) * (reward - Q_a[action]) 

            for possible_action in range(arms):
                gradient = reward - Q_a[possible_action] 
                if possible_action == action:
                    H_a[possible_action] = H_a[possible_action] + alpha * gradient * (1. - pi_a[possible_action])
                else:
                    H_a[possible_action] = H_a[possible_action] - alpha * gradient * (pi_a[possible_action])

            pi_a = softmax(H_a)
            rewards[run][step] = reward
            optimal_actions[run][step] = 1 if action == bandit.get_optimal_action() else 0

    plt.subplot(121)
    plt.plot(np.mean(rewards, axis=0), label='alpha={}'.format(alpha))
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Average reward', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    plt.subplot(122)
    plt.plot(np.mean(optimal_actions, axis=0), label='alpha={}'.format(alpha))
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('% Optimal action', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)

    plt.show()

if __name__ == "__main__":
    # stationary_env()
    # non_stationary_env()
    # stationary_env_UCB()
    gradient_bandit()