from mss import mss # To capture screenshots
import numpy as np
import torch
from PIL import Image
import mouse as m # Convenience and Realism in mouse inputs
from pynput import mouse, keyboard # For keylogger (and input mapping?)
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

class Thalamus():
        '''
        Many afferences and efferences passes through the thalamus, sometimes finishing there.
        Some neuron circuits finish there, in nuclei located in different regions, where new connections
        will begin.
        In some cases, like the Ventral Posterolateral Nucleus, it receives neural informations about
        limbs sensibility (fasciculus gracile and cuneate) and sends it to the brain cortex.

        Sometimes, the information can be preprocessed before being sent to the brain, which can
        be the case in hyperalgesia and hypoalgesia: the nucleus receives information about pain,
        and this information can be amplified or decreased before being passed to the brain, where
        the pain becomes conscient and can actually be felt.

        MENESES, M. Neuroanatomia Aplicada, 3 ed. Guanabara Koogan.
        '''

        def __init__(self,
                command_types = None,
                actions1 = None,
                actions2 = None,
                top = 0,
                left = 0,
                width = 1920,
                height = 1080,
                resize = None,
                reward_models = [None, None, None],
                reward_regions = [
                        (None, None, None, None),
                        (None, None, None, None),
                        (None, None, None, None)
                ]
                ):
        
                # Window resolutions for the frame grabber

                self.top = top
                self.left = left
                self.width = width
                self.height = height

                self.resize = resize # For reducing the images. Must be a tuple (Height, Width)

                # Input mappings, a list for each type

                self.command_types = command_types
                self.actions1 = actions1
                self.actions2 = actions2

                # Initializing Keyboard Controller
                # We'll use Mouse module for mouse commands

                self.k_c = keyboard.Controller()

                # Defining Reward Models -> List of Models in evaluation mode

                self.reward_models = reward_models

                # Reward regions ---> Input for Reward Models

                self.region1 = reward_regions[0]
                self.region2 = reward_regions[1]
                self.region3 = reward_regions[2]

        def grab_frame(self):

                with mss() as sct:
                        frame = sct.grab(monitor={"top": self.top, "left": self.left, "width": self.width, "height": self.height})
                        frame = Image.frombytes("RGB", frame.size, frame.bgra, 'raw', 'BGRX')

                        if self.resize:
                                frame = frame.resize(self.resize)

                        frame = np.array(frame, dtype=np.float32)
                        frame = frame/255 # Scaling images

                        frame = torch.from_numpy(frame)
                
                # ATTENTION: For some motive, .view() distorts the image
                #frame = frame.view(1, frame.size(2), frame.size(0), frame.size(1)) # (Batch, Channels, Height, Width)
                frame = frame.permute(2, 0, 1).unsqueeze(0)

                return frame
        
        def get_pseudo_prob(self, action):
                '''
                As the actions for Supervised Learning are Human-Made,
                they have to be one-hot encoded to become "log-probabilities"

                It may be interesting to test such commands as if it were a "Policy",
                a Human Policy which the RL Policy shall not diverge.
                Though it may provide a log close to -inf, thus a ratio also close to 0.
                '''

                command_type_log_prob = [1e-10] * len(self.command_types)
                action1_log_prob = [1e-10] * len(self.actions1)
                action2_log_prob = [1e-10] * len(self.actions2)

                command_type_log_prob[self.command_types.index(action[0])] = 1.0
                action1_log_prob[self.actions1.index(action[1])] = 1.0
                action2_log_prob[self.actions2.index(action[2])] = 1.0

                command_type_log_prob = torch.tensor(command_type_log_prob).unsqueeze(0) # Adding Batch = 1
                action1_log_prob = torch.tensor(action1_log_prob).unsqueeze(0)
                action2_log_prob = torch.tensor(action2_log_prob).unsqueeze(0)

                return [command_type_log_prob, action1_log_prob, action2_log_prob]
        
        def get_reward(self, screen_regions):
                '''
                "There's a clean way to define a learnable, ungameable reward. Two player games have this: +1 for a win, -1 for a loss. [...]
                Any time you introduce reward shaping, you introduce a chance for learning a non-optimal policy that optimizes the wrong objective."
                "If the reward has to be shaped, it should at least be rich.
                In Dota 2, reward can come from last hits (triggers after every monster kill by either player),
                and health (triggers after every attack or skill that hits a target.). These reward signals come quick and often." - Alex Irpan
                '''

                # One can use a Reward Model for this, which would be a Regressor or even a Classifier.
                # OpenAI prefers using the latter to avoid human bias in labeling.
                # Such strategy may be useful to provide continuous rewards

                with torch.no_grad():

                        rewardA = self.reward_models[0](screen_regions[0].to(device))
                        rewardB = self.reward_models[1](screen_regions[1].to(device))
                        rewardC = self.reward_models[2](screen_regions[2].to(device))

                # Checking if rewards is categorical
                        
                if rewardA.size(-1) > 1:
                        rewardA = rewardA.argmax()
                if rewardB.size(-1) > 1:
                        rewardB = rewardB.argmax()
                if rewardC.size(-1) > 1:
                        rewardC = rewardC.argmax()

                return rewardA + (rewardB * 2.0) + (rewardC * 3.0)
        
        def execute_command(self, command_type, action1, action2):
                '''
                Keyboard commands = ['key', key_string, None]
                Mouse commands = [button('left'/'right'), X, Y]

                Using Keyboard commands from the keylogger, but
                sticking to mouse module for realism in mouse movements
                and conveniency when selecting mouse button
                '''

                command_type = self.command_types[command_type]
                action1 = self.actions1[action1]
                action2 = self.actions2[action2]

                if "key" in command_type:

                        try:
                                self.k_c.tap(action1)

                        except: # Invalid combination
                                pass

                else:

                        try:
                                m.move(action1, action2, duration=0.1)
                                m.click(command_type)

                        except:
                                pass

                return None
        
        def capture_regions(self):
                # For the Reward Models
                # Remember that scaling is crucial
                # PIL.Image only works with uint8 arrays(integers, 0 to 255).
                # Matplotlib, for floats(type used in the models), considers 0 to 1.
                        
                with mss() as sct:

                        objective1 = sct.grab(monitor={"top": self.region1[0], "left": self.region1[1], "width": self.region1[2], "height": self.region1[3]})
                        objective1 = Image.frombytes("RGB", objective1.size, objective1.bgra, 'raw', 'BGRX')
                        objective1 = np.array(objective1, dtype=np.float32)
                        objective1 = objective1/255
                        objective1 = torch.from_numpy(objective1)
                        #objective1 = objective1.view(1, objective1.size(2), objective1.size(0), objective1.size(1))
                        objective1 = objective1.permute(2, 0, 1).unsqueeze(0)

                        objective2 = sct.grab(monitor={"top": self.region2[0], "left": self.region2[1], "width": self.region2[2], "height": self.region2[3]})
                        objective2 = Image.frombytes("RGB", objective2.size, objective2.bgra, 'raw', 'BGRX')
                        objective2 = np.array(objective2, dtype=np.float32)
                        objective2 = objective2/255
                        objective2 = torch.from_numpy(objective2)
                        #objective2 = objective2.view(1, objective2.size(2), objective2.size(0), objective2.size(1))
                        objective2 = objective2.permute(2, 0, 1).unsqueeze(0)

                        objective3 = sct.grab(monitor={"top": self.region3[0], "left": self.region3[1], "width": self.region3[2], "height": self.region3[3]})
                        objective3 = Image.frombytes("RGB", objective3.size, objective3.bgra, 'raw', 'BGRX')
                        objective3 = np.array(objective3, dtype=np.float32)
                        objective3 = objective3/255
                        objective3 = torch.from_numpy(objective3)
                        #objective3 = objective3.view(1, objective3.size(2), objective3.size(0), objective3.size(1))
                        objective3 = objective3.permute(2, 0, 1).unsqueeze(0)
                        
                return objective1, objective3, objective2