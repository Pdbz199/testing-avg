import argparse
import environments.simple_env  # Import to ensure registration
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from datetime import datetime
from models import Actor, Critic
from optimizers.adam_w_schedule_free import AdamWScheduleFree
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from utils import count_parameters


def get_optimizer(optimizer_name, params, lr, warmup_steps=200) -> Tuple[optim.Optimizer,bool]:
    if optimizer_name.lower() == "adam":
        return (
            optim.Adam(params, lr=lr, weight_decay=args.weight_decay),
            False
        )
    elif optimizer_name.lower() == "adamw_schedule_free":
        return (
            AdamWScheduleFree(
                params,
                lr=lr,
                warmup_steps=warmup_steps,
                weight_decay=args.weight_decay,
            ),
            True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train agent with specified optimizer')
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'adamw_schedule_free'],
                      help='Optimizer to use (default: adam)')
    parser.add_argument('--warmup_steps', type=int, default=200,
                      help='Warmup steps for AdamWScheduleFree (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                      help='Learning rate (default: 0.0003)')
    parser.add_argument('--use_tensorboard', action='store_true',
                      help='Enable tensorboard logging')
    parser.add_argument('--max_grad_norm', type=float, default=float('inf'),
                      help='Maximum gradient norm for clipping (default: inf)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                      help='Weight decay coefficient (default: 0.0)')
    args = parser.parse_args()

    # Initialize tensorboard if enabled
    if args.use_tensorboard:
        # Create a more descriptive run name with hyperparameters
        run_name = (f"{args.optimizer}_lr{args.learning_rate}_"
                   f"clip{args.max_grad_norm}_warm{args.warmup_steps}_"
                   f"wd{args.weight_decay}_"
                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        writer = SummaryWriter(f"runs/{run_name}")

        # Log all arguments as text
        arg_text = "\n".join(f"{k}: {v}" for k, v in vars(args).items())
        writer.add_text("hyperparameters", arg_text)

    # Set seeds
    torch.manual_seed(3)

    # Environment
    env = gym.make("Simple2DNavigation-v0")
    env.reset(seed=3)  # Set same seed for environment

    # Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 10
    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, hidden_dim)
    print("Actor:", actor)
    print("Actor parameter count:", count_parameters(actor))
    print("Critic:", critic)
    print("Critic parameter count:", count_parameters(critic))

    learning_rate = args.learning_rate
    actor_optimizer, needs_train_eval = get_optimizer(args.optimizer,
                                                    actor.parameters(),
                                                    lr=learning_rate,
                                                    warmup_steps=args.warmup_steps)
    critic_optimizer, _ = get_optimizer(args.optimizer,
                                      critic.parameters(),
                                      lr=10*learning_rate,
                                      warmup_steps=args.warmup_steps)

    # Put optimizers in training mode if needed
    if needs_train_eval:
        actor_optimizer.train()
        critic_optimizer.train()

    # Experiment
    num_episodes = 2000
    episodic_rewards = []
    state_paths = []

    for episode_num in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        state_paths.append([state])
        episodic_reward = 0

        while True:
            action, action_log_prob = actor(state)

            # Environment step
            next_state, reward, done, truncated, _ = env.step(action.squeeze().detach().numpy())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            done = done or truncated

            # Learning
            state_value = critic(state)
            next_state_value = critic(next_state)
            actor_objective = action_log_prob*(reward + (1-done)*next_state_value - state_value).detach()
            actor_loss = -actor_objective
            critic_loss = (reward + (1-done)*next_state_value.detach() - state_value)**2

            # Log losses if tensorboard enabled
            if args.use_tensorboard:
                global_step = episode_num * env.max_steps + len(state_paths[-1])
                writer.add_scalar('Loss/Actor', actor_loss.item(), global_step)
                writer.add_scalar('Loss/Critic', critic_loss.item(), global_step)
                writer.add_scalar('Values/State', state_value.item(), global_step)
                writer.add_scalar('Values/NextState', next_state_value.item(), global_step)

            # Actor update with gradient clipping
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            if args.use_tensorboard:
                actor_grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in actor.parameters()]))
                writer.add_scalar('Gradients/Actor_norm', actor_grad_norm.item(), global_step)
            actor_optimizer.step()

            # Critic update with gradient clipping
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            if args.use_tensorboard:
                critic_grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in critic.parameters()]))
                writer.add_scalar('Gradients/Critic_norm', critic_grad_norm.item(), global_step)
            critic_optimizer.step()

            # Log
            state_paths[-1].append(next_state)
            episodic_reward += reward

            if done:
                episodic_rewards.append(episodic_reward)
                print(f"Episode: {episode_num+1:4d}, Num steps: {len(state_paths[-1]):4d}")
                if args.use_tensorboard:
                    writer.add_scalar('Reward/Episode', episodic_reward, episode_num)
                    writer.add_scalar('Steps/Episode', len(state_paths[-1]), episode_num)
                break

            state = next_state

    # Put optimizers in eval mode if needed
    if needs_train_eval:
        actor_optimizer.eval()
        critic_optimizer.eval()

    # Close tensorboard writer if enabled
    if args.use_tensorboard:
        writer.close()

    # Plotting
    plt.plot(-100 * torch.tensor(episodic_rewards))
    plt.figure()
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red", "tab:brown"]
    for i in range(-min(30, num_episodes), 0):
        color = colors[i%len(colors)]
        state_path = torch.cat(state_paths[i])
        for i in range(state_path.shape[0]-1):
            plt.plot(state_path[i:i+2,0], state_path[i:i+2,1], alpha=(i+1)/state_path.shape[0], color=color, marker='.')
    plt.xlim([env.unwrapped.state_lower_bound[0], env.unwrapped.state_upper_bound[0]])
    plt.ylim([env.unwrapped.state_lower_bound[1], env.unwrapped.state_upper_bound[1]])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()
