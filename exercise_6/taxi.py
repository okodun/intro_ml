import time
import numpy as np
import gymnasium as gym


def start_manual_game():
    """Starts a manual game."""
    done = False
    state = env.reset()[0]  # Ensure state is captured from reset
    while not done:
        action = int(input("0/down 1/up 2/right 3/left 4/pick-up 5/drop-off: "))
        new_state, reward, done, truncated, info = env.step(action)
        state = new_state
        print(env.render())


def learn(qtable, lr=0.1, rdecay=0.9, episodes=10000, interactions=100, expl_val=0.01):
    for _ in range(episodes):
        state = env.reset()[0]
        done = False

        for _ in range(interactions):
            if np.random.uniform(0, 1) < expl_val:
                action = np.random.randint(0, qtable.shape[1])
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, truncated, info = env.step(action)

            # Update the Q-table
            update = lr * (
                reward + rdecay * np.max(qtable[new_state, :]) - qtable[state, action]
            )
            qtable[state, action] += update

            state = new_state

            if done:
                break

    return qtable


if __name__ == "__main__":
    for _ in range(10):
        r = 0
        s = 0
        for j in range(10):
            # Initialize environment
            env = gym.make("Taxi-v3", render_mode="ansi")
            env.reset()

            # Get size of state and action space
            action_size = env.action_space.n
            state_size = env.observation_space.n

            # Start learning
            qtable = learn(np.random.rand(state_size, action_size))

            # Let machine play the game
            done = False
            total_reward = 0
            steps = 0
            state = env.reset()[0]
            # print(env.render())
            for i in range(1000):
                action = np.argmax(qtable[state, :])
                new_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                state = new_state
                # time.sleep(0.5)
                # print(env.render())
                if done:
                    steps = i
                    break
            # print(f"Total Reward: {total_reward} / Steps: {steps}")
            r += total_reward
            s += steps
        print(f"Total Reward: {r / 10} / Steps: {s / 10}")
