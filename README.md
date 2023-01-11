# Hakisa - Tests Branch

This branch is just to organize the code and keep the spaghetti away from the main branch

## Currently Testing: Attention Layers

I'm not really in the field of programming as I do this for fun, but after reading papers about the state-of-the-art models in NLP, reading the RainbowDQN code and the Waveglow paper, I'm pretty sure that Attention Layers might be a technique so essential for newer models as the Conv2D layers became a few years ago.

Considering this, it might be interesting to test Attention Layers in Hakisa, mainly the MultiHead Attention used in Transformer. We can adapt its architecture in order to make it less computationally expensive(though might be less efficient) and make it assign weights to each pixel in an input frame, remarking the most relevant ones, in a process that is similar to what is done by Conv2Ds and MaxPooling in VGG19.

If everything goes alright, it might be possible to reduce a bit the amount of layers in our models, and maybe even make them more easy to handle with a personal computer without losing so much accuracy.

Currently testing this possibility for a model that generates pseudolabels for the images dataset, the Coach.

**UPDATE:** This type of layer seems promising. It has been tested for classification task with Fashion MNIST and CIFAR100 and those tests can be checked [here](https://github.com/Martyn0324/Hakisa/blob/Tests/Preprocessing/TesteAttentionLayer.ipynb). If my expectations are correct, then we could make a Feature Extractor with way less parameters, thus less computationally expensive.

## Currently Testing²: Vectorization and Softmax

The tests for vectorization before running Hakisa will continue on the main branch. Here, I'll be testing the more traditional approach for Reinforcement Learning: using Embedding layers and Softmax activation function.

I think that softmax and Embedding layers are inherently attached to computationally expensive models, as in NLP. RainbowDQN as it's implemented in SerpentAI is a quite heavy model and it uses softmax and absurd outputs sizes. Moreover, Softmax seems to be more efficient when coming after a linear neuron rather than a Conv2D, which can cause you some trouble with matrices with sizes that are too great, thus, making you deal with millions(or billions) of weights at once.

However, I think it might be worth making some tests and just trying to make the process more efficient. Reinventing the wheel didn't work for the exploration process, perhaps reinventing the wheel might not work here, again. Afterall, there might be a motive to why PhDs in computer science and AI engineering in general still rely on softmax.

The main goal stills the same: creating a RL AI that is able to run on a personal computer. If applying softmax directly to Hakisa proves to not achieve this goal, then this idea shall be discarded.

### Concept

The whole process that would be done by the vectorizer model will be done by Hakisa. Since Hakisa already has feature extraction layers, some embedding layers will be added. The output of the feature extraction and of the embedding layers will be concatenated and this result will be passed through linear layers. The output will be passed through a softmax function, generating Hakisa output.

In order to avoid big output sizes, this process will be done in separate layers according to each action, but the command type selected will still condition the action1 and action2 selected.


This method might discard the necessity of labeling each frame in your dataset, since the embedding layer will be optimized to generate vectors that provide the best loss, but only for **Play mode**. If the study mode is to be kept this way, then labeling the dataset will be necessary so Hakisa can associate states to actions.

**UPDATE:** I have discovered that what I've been refering to as "Vectorizer Model" is called, in Reinforcement Learning terminology, as "Policy", which can be a simple algorithm or a Neural Network model. This will be considered in the next updates.

**UPDATE²:**

In [the traditional approach for RL](https://github.com/Martyn0324/Hakisa/blob/Tests/ClassicRL.ipynb) I noticed that 10 steps(10 iterations) happened in less than 3 seconds. I noticed that, using softmax with action1 and action2 separately, we can have exactly 300,000 new parameters, which isn't much.
In this ClassicRL model, if we were to replace the single layer that creates a probability distribution of actions based on the list of commands by 2 layers, one that creates the probability distribution of actions1 and another that creates the probability distribution of actions2, we would have almost 33,000,000 parameters, which is considerable, but is still a light model. As a comparison, [a DCGAN Generator for the CelebA dataset](https://medium.com/analytics-vidhya/dcgan-tutorial-by-aniket-maurya-f6c137ef0053) might have around 12,700,000 parameters, and it takes around 0.5 seconds in a GTX 1650 Ti to make a full iteration.

Thanks to this separation between action1 and action2, it's not the softmax(or, more precisely, the Policy Probability Distribution) that will make Hakisa transform your GPU into a barbecue, it's actually the feature extracting layers, which were made based on VGG19, following the pattern Conv2D -> Conv2D -> MaxPool2D.
If we were to optimize Hakisa and make her less computationally expensive, we must focus on those layers.

**Update³:**

https://user-images.githubusercontent.com/28028007/211701093-b4c789c0-cac4-4a2d-8131-6e7e9a5dcdc7.mp4

This is the result for the Successful Prototype after training for 120 steps(5 minutes). The reward consisted of simply the game score passed to a log10.
One can notice that Hakisa learned more or less that she must avoid the bullets, and that if she uses her bombs she kills all enemies, thus gets easy points.

The complete code can be found [here](https://github.com/Martyn0324/Hakisa/blob/Tests/SuccessfulPrototype.ipynb) with each cell output recorded, so you can see more or less what happens.

## Possibility: Self-learning and Reinforcement Learning

Getting data for Hakisa is quite fun: all you have to do is play a game with a screengrabber on. However, labeling those images is quite tough. It's time consuming, boring and might attract some tendonitis... For this motive I was trying to make a good model to label games screenshots.

However, it might also be interesting to do another thing: the labeler model is more or less a classifier, it extracts features from the input, and then assigns a label to it. While Hakisa is also a classifier, extracting features from the input and assign a vector/label to it. So, why not mix both?
In `Self_Learning.ipynb`, there's a code to test this. There's a feature extractor network with some dropout to add randomness, a teacher network which will generate pseudo-labels and there's Hakisa which will attach those two and also generate commands.

The loss for the teacher model is usually something trying to penalize it for inconsistent outputs(which is granted by the randomness), so it's usually something like this: `loss = MSE(outputA, outputB)`. Two outputs are necessary, so 2 iterations. For this, the teacher model is usually a small network.

Also, the teacher network would be used exclusivelly during Study Mode, since it's the mode that requires Hakisa to output corresponding labels.

This architecture would discard the Exploration Mode and change quite a bit the Study Mode. It would also require to use less hyperparameters in the layers(such as less feature maps generated by the Conv2D layers), since things will get heavier naturally. Play Mode stills the same.

Both softmax and vectors might be useful in this architecure, there's just the bothering part that vectors would still require a vectorizer model. I still don't see how this model could be attached to Hakisa.


**UPDATE:** ChatGPT, the new trend, uses a Reward model to give Rewards to GPT-3. This model is actually [trained using Supervised Learning in order to provide rewards to GPT-3](https://arxiv.org/pdf/2203.02155.pdf), which, after being fine-tuned, is trained using Reinforcement Learning and PPO.
We might be able to use [a Self-Learning + Supervised Fine-tuning reward model](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/#combined-with-powerful-pre-training) to provide rewards to Hakisa. We might even attach this model to Hakisa itself.

In this case, the reward model would be trained in a similar way than the teacher model. It would be penalized for inconsistent outputs during the Self-Learning phase, and, during the Supervised Fine-tuning, it would have its weighted tuned by MSE(predicted_reward, reward_label).

If we don't attach this model to Hakisa, it could be in evaluation mode during the Reinforcement Learning stage.

However, if we do attach it to Hakisa, in order to avoid the possibility of "cheating" [Hakisa giving rewards to herself when she actually doesn't deserve it](https://www.deepmind.com/blog/specification-gaming-the-flip-side-of-ai-ingenuity), we could make the reward model in evaluation mode during most of the Reinforcement Learning stage, with it backing optimized only after a N number of steps, as Hakisa performance changes (if everything goes right, for the better). Though I don't know exactly how this backpropagation could work.
