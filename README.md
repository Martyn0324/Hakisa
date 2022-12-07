# Hakisa - Tests Branch

This branch is just to organize the code and keep the spaghetti away from the main branch

## Currently Testing: Vectorization and Softmax

The tests for vectorization before running Hakisa will continue on the main branch. Here, I'll be testing the more traditional approach for Reinforcement Learning: using Embedding layers and Softmax activation function.

I think that softmax and Embedding layers are inherently attached to computationally expensive models, as in NLP. RainbowDQN as it's implemented in SerpentAI is a quite heavy model and it uses softmax and absurd outputs sizes. Moreover, Softmax seems to be more efficient when coming after a linear neuron rather than a Conv2D, which can cause you some trouble with matrices with sizes that are too great, thus, making you deal with millions(or billions) of weights at once.

However, I think it might be worth making some tests and just trying to make the process more efficient. Reinventing the wheel didn't work for the exploration process, perhaps reinventing the wheel might not work here, again. Afterall, there might be a motive to why PhDs in computer science and AI engineering in general still rely on softmax.

The main goal stills the same: creating a RL AI that is able to run on a personal computer. If applying softmax directly to Hakisa proves to not achieve this goal, then this idea shall be discarded.

### Concept

The whole process that would be done by the vectorizer model will be done by Hakisa. Since Hakisa already has feature extraction layers, some embedding layers will be added. The output of the feature extraction and of the embedding layers will be concatenated and this result will be passed through linear layers. The output will be passed through a softmax function, generating Hakisa output.

In order to avoid big output sizes, this process will be done in separate layers according to each action, but the command type selected will still condition the action 1 and action2 selected.


This method might discard the necessity of labeling each frame in your dataset, since the embedding layer will be optimized to generate vectors that provide the best loss, but only for **Play mode**. If the study mode is to be kept this way, then labeling the dataset will be necessary so Hakisa can associate states to actions.
