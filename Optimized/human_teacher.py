import torch
from pynput import mouse, keyboard # For keylogger (and input mapping?)

# Human Teaching

'''
"In SL training, we found a learning rate of 1e-4 and 10 training epochs achieve the
best result. The best model achieves a 0.15 win rate against the level-1 built-in AI. Note
that though this result is not as good as that we acquire in the HRL method, the training
here faces 564 actions, thus is much difficult. The 1e-4 learning rate is also selected by
experiments and is different from the default 1e-3 in the AlphaStar pseudocodes. We find
that training more than 10 epochs will easily fall in overfitting, making the agent can't do
any meaningful things." - Liu, Ruo-Ze et al. On Efficient Reinforcement Learning for Full-length Game of StarCraft II

"There is an important connection between the optimal action-value function Q^*(s,a) and the action selected by the optimal policy."
"The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next."

Humans don't learn immediately, it's necessary some time for that.
https://en.wikipedia.org/wiki/Learning

The mechanism is quite similar for memory. It's necessary that the neurons synthetise proteins, especially NMDA receptors.

https://en.wikipedia.org/wiki/Memory_consolidation
https://en.wikipedia.org/wiki/Henry_Molaison

"Like most people performing this task for the first time, he did not do well and went outside the lines about 30 times.
H.M. made about 20 errors on the second trial, 12 errors on the third, and by the 10th trial on the first day he only made about 5-6 errors.
Each time H.M. performed the task, he improved even though he had no memory of the previous attempts or of ever doing the task."

In this case, we could consider that the human inputs come from an optimal policy.
When a human plays a game, they try to take an action that provides the greatest reward possible in the next state.
Thus, the value for each state will simply be the reward + the next reward.
'''

class Keylogger():

    def __init__(self):

        self.keyboard_inputs = None
        self.mouse_clicks = None

    # Functions for keylogger in order to get actions
    # Rather than relying solely on randomness with Exploration,
    # we'll teach her how to play through Supervised Learning

    # The event listener for the keyboard
    def on_press(self, key):
        try:
            self.keyboard_inputs.append(['key', key.char, None])  # If the key has a printable representation
        except AttributeError:
            self.keyboard_inputs.append(['key', str(key), None])  # For keys like 'ctrl', 'shift', etc.

    # The event listener for the mouse
    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            button = 'left'
        else:
            button = 'right'

        if pressed:
            self.mouse_clicks.append([button, x, y])

    def register(self):

        # IMPORTANT: THESE **MUST** be lists.
        # The listeners can't handle variable assignment.

        self.keyboard_inputs = []
        self.mouse_clicks = []

        kl = keyboard.Listener(on_press=self.on_press)
        ml = mouse.Listener(on_click=self.on_click)

        kl.start()
        ml.start()

        while self.keyboard_inputs == [] and self.mouse_clicks == []:

            pass

        if self.keyboard_inputs == []:

            action = self.mouse_clicks[0]

        else:

            action = self.keyboard_inputs[0]

        return action

def record(central, keylogger):

    # Collecting State

    obs = central.grab_frame()

    # Initializing Keylogger

    action = keylogger.register()

    # Collecting Action Probability

    prob = central.get_pseudo_prob(action)

    # Collecting consequences

    next_observation = central.grab_frame()
    obs_rewards = central.capture_regions()

    reward = central.get_reward(obs_rewards) # It's only necessary to use immediate reward.

    return obs, next_observation, action, prob, reward

def save_chunk(data, chunk_number, human_path):

    # Processing colected observations
    # Unfortunately, the save function tends to slow down the computer.

    chunk = torch.cat(data, 0)
    torch.save(chunk, f'{human_path}/human_chunk_states{chunk_number}.pt')

    return None