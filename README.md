# Hakisa
SerpentAI's Awesome, but, unfortunately, it's too expensive computationally. Let's see what we can do about that

The idea of SerpentAI, when using Reinforcement Learning algorithms, is basically grab screenshots in real time, pass those screenshots to an algorithm and, then, perform a command according to that algorithm's output.

However, unfortunately, Serpent also relies on many, MANY intermediaries, which makes it too slow to execute each step(takes some minutes to start a run, each step in random mode takes around 3~4 seconds and, in my case, I simply can't use training mode in my personal computer with a GTX 1650 Ti).

By removing those intermediaries(at least during training) and by having greater access and control over the algorithm we're going to use, it's possible to diminish the time between steps to around 1.3 seconds.

## General

Initially, we'll use no dataset. The only thing we're going to use is an input map. This input map is a list of commands in a specific structure `('command_action')` which will be used, in the Pytorch's Dataset class in order to generate a dictionary, where each input map(key) will be assigned to a value between -1 and 1.

We will, then, use a loop to make our neural network, Hakisa, play a game which will be our active window. With each step, a screenshot will be taken in real time and passed as input to Hakisa, which will then generate an output accordingly.

Since Neural Networks can only generate floats, the outputs generated by Hakisa will be passed to a K-Nearest Neighbors that has been fitted to that dictionary values in order to get the value that is closest to Hakisa's output. With that value, we can get the dictionary key that will, then, be used to execute a command through PyAUTOGUI module.

Hakisa will have 3 learning modes: exploration, study and play.

### Exploration Mode

Hakisa will simply play the game and generate outputs according to the images she's receiving. Each step will generate a memory for Hakisa(that will actually be part of the Dataset class, not Hakisa class) composed of the input frame, the output key in the input mapping, its value and the reward obtained. This memory will, posteriorly, be our dataset for the study mode.

At each step, a tuple `(frame, key, value, reward)` will be added to the memory. If the memory gets full(as defined when initializing Dataset class), the items with lower rewards will be discarded in order to add the new items.

In this mode, it's important that Hakisa generates outputs as diverse as possible, so avoid using weights initialization through normal or uniform distribution.

**EDIT: Now using random noise as input during this mode. This way, we can get a wider range of outputs and, thus, expose Hakisa to a wider range of situations**

### Study Mode

We'll use Hakisa's memory to generate a classic dataset for machine learning. Each frame will serve as input, and each value, a label. As criterion, we'll use MSE Loss.

This stage works as a way to make Hakisa identify patterns in each situation in the game and associate those situations with a value(controller command) that is best suited for a situation. The input is a frame where a projectile is coming towards your character? Then the best output is the one associated with the command "move left" or "move right". There's an enemy in the input frame? The best output is the one associated with the command "shoot".

Of course, not necessarily Hakisa's memory will contain exactly the best output for that situation. For this motive, it's important to use a memory size smaller than the number of exploration steps. This way, the memory will tend to have the best outputs for specifics situations.

You can also use a ready-made dataset, with frames captured by you when you were playing and labels defined by yourself(maybe there's some way to properly capture mouse/keyboard commands in real time...)

### Play Mode

Here, Hakisa will use what she learned in the previous stages and play all by herself. At each step, a screenshot will be taken and passed to Hakisa as input, and she'll generate an output accordingly.

We're gonna be using a custom loss function, GameplayLoss, which will use the reward generated by Hakisa's command in order to get gradients for backpropagation. This function is log based so the gradients returned are bigger as the reward decreases, and smaller as the reward increases.

This way, Hakisa can correct some associations she made in the study mode, probably because she didn't get exposed to determined situations or because she didn't generate the best output for that situation during the exploration mode.

This way, Hakisa can play and get better and better as she plays, all by herself.

**EDIT: The GameplayLoss function is actually making Hakisa's outputs as random as when she's on Exploration mode. Possible corrections might be using cumulative rewards during exploration mode or simply remaking/replacing this function. Softmax and Cross-Entropy Loss must be avoided in order to avoid great output sizes.**

**Also good consideration for gameplay loss function: Liu, Ruo-Ze et al. Rethinking of AlphaStar, page 8: https://arxiv.org/pdf/2104.06890.pdf**
