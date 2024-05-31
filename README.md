
# Reinforcement Learning

This project implements a Q-learning scenario where a Sender agent communicates with a Receiver agent to navigate a grid and find a prize. The project includes the following components:

- Sender and Receiver agents implementing Q-learning.
- Multiple grid configurations
- Various learning scenarios to evaluate training performance
- Functions to run episodes and update Q-values.
- Animation of the training process with a progress bar.

The reinforcement learning training is also illustrated by creating an animation of the sender and receiver using cartoonic characters. The bee acts as the receiver of the signal and the flower acts as the sender of the signal. Watch the bee and the flower learning how to communicate in order to get the reward.

<iframe width="560" height="315" src="(https://youtu.be/VyvJmquF8o4)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
[Bee and flower Adventure](https://youtu.be/VyvJmquF8o4)

## Project Structure

- `Sender` class: Implements the Q-learning sender agent.
- `Receiver` class: Implements the Q-learning receiver agent.
- `get_grid(grid_name)`: Function to generate predefined grid configurations.
- `legal_move(posn_x, posn_y, move_id, grid)`: Function to determine the legality of a move on the grid.
- `run_episodes(sender, receiver, grid, num_ep, delta, ep_frame_freq)`: Runs the reinforcement learning scenario for a specified number of episodes.
- `plot_sender_policy(sender)`: Plots the sender's policy.
- `plot_receiver_policy(receiver, signal_num)`: Plots the receiver's policy.
- `animate_sender_receiver(frames, grid, savepath, fps)`: Creates an animation of the training process.
- `run_with_animation()`: Runs the entire process including training and animation.

## Installation

Ensure you have the following packages installed:

- `numpy`
- `matplotlib`
- `tqdm`

You can install these packages using pip:

```bash
pip install numpy matplotlib tqdm
```

## Usage

1. **Define Parameters**: Set parameters such as the number of episodes, grid type, and learning rates.
2. **Initialize Agents**: Create instances of `Sender` and `Receiver`.
3. **Run Episodes**: Use `run_episodes` to train the agents.
4. **Generate Animation**: Use `animate_sender_receiver` to create and save the animation of the training process.
5. **Plot Policies**: Optionally, use `plot_sender_policy` and `plot_receiver_policy` to visualize the learned policies.

## Example

```python
def run_with_animation():
    grid_name = 'fourroom'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    alpha_init = 0.9
    alpha_final = 0.01
    num_ep = 1000
    epsilon = 0.1

    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_ep, epsilon, discount)

    reward_vals, receiver, sender, frames = run_episodes(sender, receiver, grid, num_ep, delta, ep_frame_freq=100)
    animate_sender_receiver(frames, grid, fps=5)

if __name__ == "__main__":
    run_with_animation()
```

## Contributors

This project is open to contributions. Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.
