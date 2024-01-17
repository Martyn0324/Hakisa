# Hakisa
SerpentAI's Awesome, but, unfortunately, it's too expensive computationally. Let's see what we can do about that

The idea of SerpentAI, when using Reinforcement Learning algorithms, is basically grab screenshots in real time, pass those screenshots to an algorithm and, then, perform a command according to that algorithm's output.

However, unfortunately, Serpent also relies on many, MANY intermediaries, which makes it too slow to execute each step(takes some minutes to start a run, each step in random mode takes around 3~4 seconds and, in my case, I simply can't use training mode in my personal computer with a GTX 1650 Ti).

By removing those intermediaries(at least during training) and by having greater access and control over the algorithm we're going to use, it's possible to diminish the time between steps to around 1.3 seconds.

## Summary/Résumé

Reinforcement Learning consists essencially on the use of a function with optimizable parameters that will be optimized towards a specific task through the feedback it receives while executing such task.

A function with optimizable parameters can be viewed as an Artificial Intelligence, which can be, then, a model like Linear Regression, Decision Tree, KNN, etc. When using Neural Networks, Reinforcement Learning can bend with Deep Learning, categorizing Deep Reinforcement Learning.

There are some problems around the terminology in Reinforcement Learning due to its specificity. First of all, the task that is going to be executed is divided into states. For each state, there are many possible actions, as well as it's possible to assign a value for each state (some states are more desirable than others).

Out of many methods of implementing Reinforcement Learning, there are 2 main cathegories: Policy Methods and Action-Value Methods:

* Policy Methods: Uses 2 functions, a Policy Function to generate actions, and a Value Function to determine the value for the next state.
* Action-Value: Uses a single function which is both Policy and Value Function. The idea is that the function will determine a value for each possible action that can be chosen for a given state. The action with highest value tends to be chosen (`.argmax()`)

When implementing a Policy Function and a Value Function, one can unify both to make an Actor-Critic function, generating 2 different outputs: the probability of each action (policy) and the value prediction.

The value of each state is often based on the reward function, which provides a feedback for the model.

However, one big problem Reinforcement Learning has is its instability. In fact, it's often difficult to reproduce results, even when using the same hyperparameters, reward and environment. Not only that, but it's also hyperparameter sensitive. For these motives, it's quite common to avoid training the model online - when the model both interacts and learns from the environment - and prefer to train it offline - that is, the model collects data from the environment by interacting with it, then, after this interaction if finished, the model is submitted to a training process similar to Supervised Learning.

### Optimization Methods

Offline training is what is commonly used with Proximal Policy Optimization (PPO), developed by OpenAI - the most proeminent enterprise in the field of RL -. During exploration, the data collected provides information on the environment state, the action the policy model executed, the reward obtained and the predicted value for the next state. When the exploration is finished, the policy and value models are submitted to a "replay", receiving each collected state as input and generating their respective outputs. The value model is penalized when it diverges from predicting the sum of the next state value with the reward value. The policy, however, is trained with a surrogate loss that tries to push it a bit away from generating the same outputs, while also constraining and penalizing it for diverging too much from the previous policy. In PPO2, an early-stopping criterion is used to avoid making the policy becoming too divergent, that is the KL-Divergence between the current policy output and the previous policy output.

The final loss will be, then, a weighted sum between the surrogate loss (which tends to have a higher weight), the value loss (which is a Mean Squared Error) and the entropy loss (which penalizes the policy for generating low diversity outputs). This aims to avoid both the situation where the model is always generating the same output and the situation where it gets completely random.

**However, my experiments in Street Fighter II showed that, contrary to the expected, PPO is quite prone to generating exactly what is aims to avoid. Not only the method is much more slow and computationally expensive due to the tons of mathmatical operations it requires, it also forces the model to "not learn too much", which makes absurdly big datasets and training time a must. Not only that, but you may also just lose your time creating a big dataset, as the early-stopping may make the model stop training right at the beginning of the process. Theorically, this avoids model collapse or performance drop, but the difficulty encountered while trying to find the perfect hyperparameters for PPO makes it a bit dubious...**

Unlike PPO, Q-Learning is the main technique used for Action-Value method and, though it's a bit old, rumours suggests that it has been recently discovered by OpenAI and [may be used to](https://arxiv.org/pdf/2102.04518.pdf) [train GPT-5](https://arxiv.org/pdf/2305.20050.pdf). Q-Learning is commonly represented by tabular search, where each specific state is mapped to a specific value by a model. Sometimes the value is the reward itself, and the model may not even be an optimizable function at all, being simply a bot that interacts with the environment and consults the table to see how it's going.

However, it's also possible to implement Q-Learning through the use of Machine Learning models. The most known case is Deep Q-Learning Network (DQN), which uses a Neural Network to map both the possible value for each state. This can be interpreted as mapping the value for each action given a certain state. With that, in order to extract the best action for that state, we just have to extract the value index.

I took some time to understand that. But, once you do, DQN gets much more intuitive and easier to implement. It doesn't require thousands of mathmatical operations to be done manually. The only excentric element is the sampling schedule, like epsilon-greedy, which is a method that randomly samples an element from the output generated by the model in order to generate a random action and make the states collected more diverse, thus helping during learning. Besides that, all one needs to do is make your model generate its outputs, extract the actions to apply on the environments, and compare the model outputs with the same outputs summed together with the reward.

Actually, there's another technical detail: it's good to use a target network, which is a copy of the model, to generate the values that will be summed with the reward and act as targets from the loss (MSE). This provides greater stability as the target network may have its optimization delayed by some iterations.

**My experiments with Street Fighter II showed that, despite the spotlights lying on PPO, this method is much easier to tune, less computationally expensive (unless your model is too big) and much faster. It also seems more resilient to failure modes, but it also seems more sensitive to the reward function. Using discrete Rewards (rewards given only after a big event happens in the game, like win or lose) may be greatly troublesome here. I recommend sparing some time to [train reward models](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)**


Finally, there's also the method where one can use Genetic Algorithms to select the best policy network for the given environment. I'm pretty fond of GAs, but unfortunately creating multiple copies of a Neural Network, applying mutations to their parameters and making complete iterations through the whole dataset with each one of them is too computationally expensive and too slow. However, it may also be a considerable option when using shallow models and as a preliminary stage to DQN.

## General

Given the previous text, it's easier to trace a method to implement Hakisa.

We will use an input mapping divided into 3 types of actions: `command_type`, `action_1` and `action_2`. This cathegorization is mandatory in order to make working with mouse inputs easier and avoid memory issues, since in a screen with 1920x1080 dimensions, there are 1920 X coordinates and 1080 Y coordinates for mouse positioning, thus 2,073,600 different possibilities of outputs. It's easier to blow up your memory with such output size, and using a bottleneck layer may cause great information loss. By assigning 1920 X possibilities to an action 1 and 1080 Y possibilities to action 2, we don't lose our 2 million total possibilities, but we can now work with 2 different layers that will have to deal with much less data, and the need for information compression will be much lower - If we were to use a bottleneck layer that provided an output with size 1, and this output was passed to an linear layer which provides an output 2,073,600, we would have almost the same number of parameters as using one layer with input 512 and output 1920 + one layer with input 1024 and output 1080.

In this case, `command_type` will determine wheter the action is a keyboard command, a mouse movement, right click or left click. `action_1` stands for the keyboard key to be pressed (as a string) or the X coordinate for the mouse input, while `action_2` stands for the Y coordinate for mouse input, or simply `None` for keyboard input.

Hakisa will receive as inputs the frame of the game she must play, and will generate multiple outputs for each action type. Since she will be a Deep Q-Learning Network, each output for each action type corresponds to a value for an action (so no activation functions will be used in the final layers). This action can be extracted by using the index of that value, and this index can be used in the input mapping list (also divided into three action types) to extract the action string that will be performed.

Hakisa will be used to play complex games and, as such, we can expect a hard time with instability. For this motive, her training will be divided into 5 stages:

* **Human Teacher Phase** : The human will play the game, with the state (game frame), actions and rewards being stored. This data will be used to create a dataset. The human actions will also be one-hot encoded in order to be used as labels in the next phase
* **Study Phase**: Hakisa will be trained in a Supervised Learning configuration. Each state from the dataset will be passed to her, and the outputs will be compared with the human actions which will work as labels. The idea is to make her try to mimetize the human gameplay, which may make her acquire an average or, at least, a beginner skill level on the game.
* **Exploration Phase**: Hakisa will play the game by herself, generating new data for the next stage.
* **Consolidation Phase**: The acquired data (game frames and rewards) will be passed to Hakisa again and to the target network, optimizing Hakisa in a Reinforcement Learning manner.
* **Play (and learn?)**: Hakisa plays the game freely in evaluation mode. One can also collect new data here to train her again posteriourly.

Note that Reinforcement Learning is actually applied as a fine-tuning to the Supervised Learning method. This is used to make the process less unstable (note that even in OpenAI's GPT-4, Reinforcement Learning was applied to fine-tune GPT-3. The same happened to OpenAI's Five in Dota 2, to DeepMind's AlphaStar in StarCraft 2 and to Ruo-Ze Liu's HierNet, also in StarCraft 2).

The Exploration Phase may be an optional stage, with one possibly using the human dataset for Reinforcement Learning in Consolidation Phase. However, that may not be productive if the same data has already been used in Supervised Learning.

One may also consider to unify Exploration and Consolidation Phase into a single Play and Learn phase, but that may also be too prone to unstability (it was already troublesome in Street Fighter II).


### UPDATE - PPO2 IMPLEMENTATION USING GYM RETRO ENVIRONMENT

https://github.com/Martyn0324/Hakisa/assets/28028007/7b2da36c-1675-4f11-922c-72cfeff6b222

This video was recorded after an exploration phase with 1000 steps, which means the model was trained with 1000 samples.
The training ocurred during 10 epochs without early-stopping as its criterion wasn't reached (KL Divergence between policies greater than 0.05).

Such low number of samples, together with the low diversity of the states got in exploration phase (which indeed consisted basically of Chun Li standing still, attacking randomly),
justify the model's behaviour.

### UPDATE² - DEEP Q-LEARNING IMPLEMENTATION USING GYM RETRO ENVIRONMENT

https://github.com/Martyn0324/Hakisa/assets/28028007/90dd31ea-cbf9-4cde-a385-de0f74c4fd60

This one was also recorded after an exploration phase with 1000 steps, which means the model was trained with 1000 samples, during 10 epochs.
No early-stopping was used under any criterion.

The reward function was customized (that is, the original reward given by Gym was replaced by a custom one),
another exploration phase was ran for 1000 steps.
After that, the consolidation phase was conducted for 2 epochs.
This was the result:

https://github.com/Martyn0324/Hakisa/assets/28028007/ba028a2c-5b6b-4857-8d7a-71d65295ab20

### UPDATE³ - EVOLUTIONARY ALGORITHM ON REINFORCEMENT LEARNING

https://github.com/Martyn0324/Hakisa/assets/28028007/ef8d9c11-2e4a-4800-90d6-679579799b7f

This one was optimized using evolutionary algorithms, from a population with 20 different models.
The population was evaluated for 5 generations/epochs, and after each generation, the 10 most fit were selected
to duplicate, eliminating the 10 less fit. Each generation was composed of 1000 iterations.

The number of mutations decreased as the generation increased. The fitness function was a custom reward fuction
with all rewards obtained through a generation multiplied by a discount (uncertainty) factor.


### References

[Huang, Shengyi et al. The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/): This blog is simply **glorious**

Ruo-Ze Liu's implementation of PPO2: https://github.com/liuruoze/HierNet-SC2/blob/main/algo/ppo.py

[Liu, Ruo-Ze et al. An Introduction of mini-AlphaStar](https://arxiv.org/pdf/2104.06890.pdf)

[Liu, Ruo-Ze et al. On Efficient Reinforcement Learning for Full-length Game of StarCraft II](https://arxiv.org/pdf/2209.11553.pdf)

[Liu, Ruo-Ze. Rethinking of AlphaStar](https://arxiv.org/pdf/2108.03452.pdf)

[Yang Yu, Eric. Coding PPO From Scratch With Pytorch](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8)

[Reinforcement Learning Classes from Didática Tech(PT-BR)](https://didatica.tech/curso-aprendizado-por-reforco-algoritmos-geneticos-nlp-e-gans/)

[Weng, Lilian. A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) - How could I forget Lil's Log? Especially since she's a Research Leader from OpenAI...

[OpenAI's Spinning Up, by Joshua Achiam](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) - The introduction is wonderfully awesome for catching up with terminology (seriously, maybe even folks from the field mix things up). It's also full of interesting hyperlinks.

[Irpan, Alex. Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) - A comforting post on why your Deep RL model may not be working. The problem is not always between the screen and the chair. However, this piece is also a Deep Reinforcement Learning model...a biological one, and it may be able to learn and try some tricks to make the silicon model have a higher chance of working.

[Fish, Amid. Lessons Learned Reproducing a Deep Reinforcement Learning Paper](https://amid.fish/reproducing-deep-rl)

~~ChatGPT...despite some misinformation~~ - Seriously, don't use ChatGPT. Not even Bing's. At most, ask it to summarize texts on this subject.
