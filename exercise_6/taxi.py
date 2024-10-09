import time
import numpy as np
import gymnasium as gym


def start_manual_game():
    """starts a manual game"""

    done = False
    while not done:
        action = int(input("0/down 1/up 2/right 3/left 4/pick-up 5/drop-off: "))
        _, _, done, _, _ = env.step(action)
        print(env.render())


class Agent:
    """class for agent that utilizes Q-Learning"""

    def __init__(self, env, lr=0.1, rdecay=0.9, e=10000, i=100, exp=0.01):
        """creates new Agent object"""

        self.ENVIRONMENT = env
        self.LEARNING_RATE = lr
        self.REWARD_DECAY = rdecay
        self.EPISODES = e
        self.INTERACTIONS = i
        self.EXPLORE_VALUE = exp
        self.QTABLE = None

    def learn(self):
        """learns optimal behavior in evironment"""

        # initialize Q-Table
        qtable = np.random.rand(
            self.ENVIRONMENT.observation_space.n, self.ENVIRONMENT.action_space.n
        )

        # start iteration over episodes
        for _ in range(self.EPISODES):

            # define environment and set done flag
            state = self.ENVIRONMENT.reset()[0]
            done = False

            # start interacting with environment
            for _ in range(self.INTERACTIONS):

                # randomly explore environment
                if np.random.uniform(0, 1) < self.EXPLORE_VALUE:
                    action = np.random.randint(0, qtable.shape[1])
                else:
                    action = np.argmax(qtable[state, :])

                # execute action
                new_state, reward, done, _, _ = env.step(action)

                # update Q-table value
                update = self.LEARNING_RATE * (
                    reward
                    + self.REWARD_DECAY * np.max(qtable[new_state, :])
                    - qtable[state, action]
                )
                qtable[state, action] += update

                # set new state
                state = new_state

                # check if solution was found
                if done:
                    break

        # save Q-Table
        self.QTABLE = qtable

    def evaluate(self, max_range=1000, visual=False):
        """evaluates the found solution"""

        # initialize result variables
        total_reward = 0
        total_steps = 0

        # reset environment
        state = self.ENVIRONMENT.reset()[0]

        # show gameplay if visual is set to true
        if visual:
            print(env.render())

        # start playing the game
        for step in range(max_range):

            # choose action and execute it
            action = np.argmax(self.QTABLE[state, :])
            new_state, reward, done, _, _ = env.step(action)

            # show gameplay if visual is set to true
            if visual:
                time.sleep(0.5)
                print(env.render())

            # update reward and state
            total_reward += reward
            state = new_state

            # break if done
            if done:
                total_steps = step
                break

        return total_reward, total_steps


if __name__ == "__main__":

    # define number of iterations
    N = 10

    # initialize result variables
    total_reward = 0
    total_num_of_actions = 0

    # start iterations
    for _ in range(N):

        # initialize environment
        env = gym.make("Taxi-v3", render_mode="ansi")
        env.reset()

        # create agent and learn
        agent = Agent(env)
        agent.learn()

        # evaluate and save results
        evaluation = agent.evaluate()
        total_reward += evaluation[0]
        total_num_of_actions += evaluation[1]

    # print result
    result_string = f"""
    The agent achieved an average reward of {total_reward / N:.2f}.
    On average, it took {total_num_of_actions / N:.2f} steps to solve each problem.
    The agent successfully solved {N} different problems.
    """
    print(result_string)
