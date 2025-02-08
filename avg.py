import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from models import Actor, Critic
from utils import count_parameters


if __name__ == "__main__":
    # Problem
    torch.manual_seed(3)
    state_lower_bound = torch.tensor([[-1., -1.]])
    state_upper_bound = torch.tensor([[1., 1.]])
    action_lower_bound = 0.1*torch.tensor([[-.1, -.1]])
    action_upper_bound = torch.tensor([[.1, .1]])
    dt = 1

    # Agent
    state_dim = 4
    action_dim = 2
    hidden_dim = 10
    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim+action_dim, hidden_dim)  # Represents Q function
    print("Actor:", actor)
    print("Actor parameter count:", count_parameters(actor))
    print("Critic:", critic)
    print("Critic parameter count:", count_parameters(critic))
    learning_rate = 0.0003

    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=10*learning_rate)

    # Experiment
    num_episodes = 2000
    episodic_rewards = []
    state_paths = []
    i = 0
    for episode_num in range(num_episodes):
        position = torch.rand((1, 2)) * (state_upper_bound-state_lower_bound) + state_lower_bound
        velocity = torch.zeros((1, 2))
        state = torch.cat((position, velocity), axis=1)
        state_paths.append([state])
        episodic_reward = 0
        while True:
            action, action_log_prob = actor(state)

            # Receive reward and next state
            position += velocity*dt + 0.5*action*dt**2
            velocity[position < state_lower_bound] = -0.1*velocity[position < state_lower_bound]
            velocity[position > state_upper_bound] = -0.1*velocity[position > state_upper_bound]
            position = torch.clamp(position, state_lower_bound, state_upper_bound)
            velocity += action*dt
            next_state = torch.cat((position, velocity), axis=1)
            reward = -0.01
            done = torch.allclose(position, torch.zeros(2), atol=0.25) and torch.allclose(velocity, torch.zeros(2), atol=0.1) or len(state_paths[-1]) == 4999
            
            # Learning
            state_action_value = critic(torch.cat((state, action), axis=1))
            with torch.no_grad():
                next_action, _ = actor(next_state)
                next_state_action_value = critic(torch.cat((next_state, next_action), axis=1))

            # Critic loss
            critic_loss = (reward + (1-done)*next_state_action_value - state_action_value)**2

            # Policy loss
            action, action_log_prob = actor(state, rsample=True)
            state_action_value = critic(torch.cat((state, action), axis=1))
            actor_objective = state_action_value
            actor_loss = -actor_objective

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Log
            state_paths[-1].append(next_state)
            episodic_reward += reward

            # Termination
            if done:
                episodic_rewards.append(episodic_reward)
                i += 1
                print(f"Episode: {i:4d}, Num steps: {len(state_paths[-1]):4d}")
                break

            state = next_state

    # Plotting
    plt.plot(-100 * torch.tensor(episodic_rewards))
    plt.figure()
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown"]
    for i in range(-min(30, num_episodes), 0):
        color = colors[i%len(colors)]
        state_path = torch.cat(state_paths[i])
        for i in range(state_path.shape[0]-1):
            plt.plot(state_path[i:i+2,0], state_path[i:i+2,1], alpha=(i+1)/state_path.shape[0], color=color, marker='.')
    plt.xlim([state_lower_bound[0, 0], state_upper_bound[0, 0]])
    plt.ylim([state_lower_bound[0, 1], state_upper_bound[0, 1]])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()