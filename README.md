# Hakisa
SerpentAI's Awesome, but, unfortunately, it's too expensive computationally. Let's see what we can do about that

The idea of SerpentAI, when using Reinforcement Learning algorithms, is basically grab screenshots in real time, pass those screenshots to an algorithm and, then, perform a command according to that algorithm's output.

However, unfortunately, Serpent also relies on many, MANY intermediaries, which makes it too slow to execute each step(takes some minutes to start a run, each step in random mode takes around 3~4 seconds and, in my case, I simply can't use training mode in my personal computer with a GTX 1650 Ti).

By removing those intermediaries(at least during training) and by having greater access and control over the algorithm we're going to use, it's possible to diminish the time between steps to around 1.3 seconds.

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



## General

Initially, we'll use no dataset. The only thing we're going to use is an input map. This input map is a list of commands in a specific structure `('command_action')` which will be used, in the Pytorch's Dataset class in order to generate a dictionary, where each input map(key) will be assigned to a value between -1 and 1.

We will, then, use a loop to make our neural network, Hakisa, play a game which will be our active window. With each step, a screenshot will be taken in real time and passed as input to Hakisa, which will then generate an output accordingly.

Since Neural Networks can only generate floats, the outputs generated by Hakisa will be passed to a K-Nearest Neighbors that has been fitted to that dictionary values in order to get the value that is closest to Hakisa's output. With that value, we can get the dictionary key that will, then, be used to execute a command through PyAUTOGUI module.

Hakisa will have 3 learning modes: exploration, study and play.

### Exploration Mode

Hakisa will simply play the game and ~~generate outputs according to the images she's receiving~~ she'll now generate random outputs, independently of the image she's receiving. Each step will generate a memory for Hakisa(that will actually be part of the Dataset class, not Hakisa class) composed of the input frame, the output key in the input mapping, its value and the reward obtained. This memory will, posteriorly, be our dataset for the study mode.

At each step, a tuple `(frame, key, value, reward)` will be added to the memory. If the memory gets full(as defined when initializing Dataset class), the items with lower rewards will be discarded in order to add the new items.

In this mode, it's important that Hakisa generates outputs as diverse as possible ~~, so avoid using weights initialization through normal or uniform distribution.~~ Hakisa's network won't be used at all.

### Study Mode

We'll use Hakisa's memory to generate a classic dataset for machine learning. Each frame will serve as input, and each value, a label. As criterion, we'll use MSE Loss.

This stage works as a way to make Hakisa identify patterns in each situation in the game and associate those situations with a value(controller command) that is best suited for a situation. The input is a frame where a projectile is coming towards your character? Then the best output is the one associated with the command "move left" or "move right". There's an enemy in the input frame? The best output is the one associated with the command "shoot".

Of course, not necessarily Hakisa's memory will contain exactly the best output for that situation. For this motive, it's important to use a memory size smaller than the number of exploration steps. This way, the memory will tend to have the best outputs for specifics situations.

You can also use a ready-made dataset, with frames captured by you when you were playing and labels defined by yourself(maybe there's some way to properly capture mouse/keyboard commands in real time...)

**EDIT:** Now Hakisa will also try to predict the reward she'll get in that step. This prediction will be passed to another loss having the actual reward as target.
The Study Loss will then be: study_loss = cross_entropy(command_type_output, command_type_label) + mse(action1_out, action1_label) + mse(action2_out, action2_label) + mse(predicted_reward, actual_reward).

### Play Mode

Here, Hakisa will use what she learned in the previous stages and play all by herself. At each step, a screenshot will be taken and passed to Hakisa as input, and she'll generate an output accordingly.

**EDIT:** Now, Hakisa will also receive the previous output actions and the previous reward received.

We're gonna be using a custom loss function, GameplayLoss, which will use the reward generated by Hakisa's command in order to get gradients for backpropagation. This function is log based so the gradients returned are bigger as the reward decreases, and smaller as the reward increases.

This way, Hakisa can correct some associations she made in the study mode, probably because she didn't get exposed to determined situations or because she didn't generate the best output for that situation during the exploration mode.

This way, Hakisa can play and get better and better as she plays, all by herself.

**EDIT: The GameplayLoss function is actually making Hakisa's outputs as random as when she's on Exploration mode. Possible corrections might be using cumulative rewards during exploration mode or simply remaking/replacing this function. Softmax and Cross-Entropy Loss must be avoided in order to avoid great output sizes.**

**EDIT²: The GameplayLoss function will indeed have to be remade or replaced, as its gradients makes Hakisa generate outputs that will only correspond on the extreme commands in the input mapping dictionary (she'll only generate the command for -1 and for 1).**

**Also good consideration for gameplay loss function: Liu, Ruo-Ze et al. Rethinking of AlphaStar: https://arxiv.org/pdf/2104.06890.pdf . - Still uses Categorical Cross Entropy, but might be a good inspiration.**


# Update and possible upgrades

While testing my NLP models(and also chatting in Python's Discord server) I've learned that softmax can't really be avoided. The motive is simple: numbers have a correlation between each other, but our input mappings, just like words and sentences, don't. The input map `press X` isn't bigger or smaller than `click (512, 600)`.

This can only be avoided by the use of categories. In a Classifier, for example, the number `0` can be the label `Dog`, while `1` can be `Cat`, and there's no mathematical relation between `Dog` or `Cat`, they're simply categories. This is learned with time by the classifier, through the use of a softmax or sigmoid function.
In NLP, each word is assigned to an integer, which works as a label. Letter `a` can be the label `0`, `b` be `1`, `c`, `2` and so on. This relation comes up as the model trains. The same happens with Reinforcement Learning models, like Rainbow DQN, or even the Hierarchical AlphaStar.

However, there's a way to avoid having to use softmax, despite this...kinda. In NLP, it's used the technique `word2vec`, which converts words to vectors, that is, a single value. `word2vec` in fact, consists on the use of algorithms to associate words to specific values and it's included in the embedding layers, commonly used in NLP models and it's used in the mentioned paper above. It makes the word `apple` has a vector closer to `fruit` than to, let's say, `car`.

For Hakisa, we could do something like that to make associations between certain input mapping and a vector. `press X` can be associated with the number `0.75` and `press Z` can be associated with `0.76`, while `click (512, 600)` can be associated with `0.10`.This technique, however, requires one-hot encoding and the use of softmax, which can be make things computationally expensive.
However, we could create a separate model, disconnected from Hakisa, that would be trained separately to convert each input map to a vector. After its training has been complete, it would be used to generate the dictionary of input mappings for Hakisa. After that, we'll continue making things as we do right now.


*I'll be testing this idea with NLP models and see if this works and if I should make some adjustments. Consider this text if you want to test Hakisa.*

### Vector Embedding

Applying vector embedding is promising and, differently from what I'm doing, is something that actually makes mathematical sense.
This technique is usually associated with NLP, so I think the best way to understand it for Hakisa is to relate it to NLP.

In Vector Embedding, we use a quite shallow network(or maybe some other simple model) to extract context from a given input and assign a vector to it. Our input can be a single word, an entire sentence or words n-grams. The closer a token vector is from another, the closer is their meaning. I suppose that the technique is more efficient with n-grams or sentences(It's hard to extract context from a single word, lost in the abyss, isn't it?).

Vector Embedding can be quite efficient in NLP when you're dealing with a reasonable amount of data(which can be understood as: you're not just playing with some dozens of words).

For Hakisa, this can be quite good for games that use mouse, since we can get up to 4, 5 or even 6 million possible commands. However, this is quite an annoyance for games that only uses the keyboard, which will have few possible inputs.
We also want to avoid directly working with millions of data in our input mapping since this will make it mandatory to use cloud servers CPUs to properly fit KNN in seconds(or minutes).

For Reinforcement Learning, we could use a similar approach. However, our input mapping actually doesn't provide any context. It's actually a feedback for a given game state, a response. So our context would be extracted from the game state, which is the captured frame.
On this case, our vectorizer model won't be able to be a simple, shallow network. We won't be able to simply make a model with a single Pytorch Embedding layer, we need to extract features from the game frame and associate them with a given command.

Also, this method might make Hakisa's Exploration mode useless, as this mode doesn't relate a game frame to a proper, "correct" command, it just tries to generate the "least worst option". Training our vectorizer based on this mode wouldn't be efficient at all.
In NLP, if the vectorizer receives the sentence "School bright keyboard Europe", it'll make the mistake of relating "school" to "bright", which is nonsense. This would be the case if we train a vectorizer model on Hakisa's random input mappings.
However, if we have a ready-made data, which could be the sentence "At school, we had computing class, but my computer keyboard was broken", our vectorizer would correctly relate the word "keyboard" to "computer", or "broken". It could relate "keyboard" to "school", but it'll also relate it to "computing" and "class".
I presume this would be the case if we used a ready-made data where in the state A, the command is properly labeled as "Jump".

For this reason, I'd recommend creating a dataset composed of game frames(which can be captured using the `mss` module. It doesn't lag your game, I promise!) and labeling each frame manually(if you can find a way to automatize this, I'd love to know it).
You don't need to capture a single frame each second as Hakisa does, but it might be important to capture a number of frames related to the probability of you being subject to each situation. Example: in Jigoku Kisetsukan, if you want Hakisa to play on harder modes, it might not be a good idea to use frames which have no bullets on the screen(and it might also be good to use frames when you're in the game over screen,hehe).

It might still be possible to use Hakisa's exploration mode to generate data for vectorization, but this would rely on using a memory_size <<<<<<< explore_steps, so it might take less time to simply create your own dataset.

Study Mode will probably have to be modified: instead of simple Supervised Learning, we'll also train a Vector Embedding Layer for each action command. So we'll have `command_type` being one-hot encoded and then serving as input for this vector embedding layer, which will throw an output with the same size as the input, which will then be passed into a Cross Entropy having the input as target. The same thing will be applied to `action1` and `action2`.

Those vector embedding layers will be used to create our input_mapping dictionary, which will now be used to convert Hakisa's outputs into input commands. KNN will then be properly fitted again.
~~*In the exploration mode, Hakisa will simply choose a random integer which will serve as an index for the list of input mappings generated upon calling the `Dataset` class*~~

Categorical Cross Entropy will be used to optimize the vector embedding layers, while Mean Squared Error will be used to optimize Hakisa's output(remember that Hakisa's output will be a vector, a single number).


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
