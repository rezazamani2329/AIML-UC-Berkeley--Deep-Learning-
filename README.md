# AIML-UC-Berkeley--Deep-Learning-

Articulate the basics of a CNN and its different layers.

Process images in preparation for training a neural network.

Train a neural network to recognize items and images.

Use a pretrained network to enhance model performance.

Interpret a neural network model.

Compare an LSTM network to time series analysis.

Train an LSTM network for time series analysis.

Apply neural network techniques to a linear regression problem.

Apply AI/ML techniques to a specific field you are interested in.Overview: Neural Networks for Vision

## Neural Networks for Vision
- A convolutional neural network (CNN) is an algorithm that processes deep learning input images by assigning importance to weights and biases for various aspects and objects in the images. Then, it works to distinguish the input images from one another. A CNN is built similarly to the connectivity pattern of neurons in the human brain and is based on the organization of the visual cortex. An individual neuron responds to stimuli only in a limited visual field region, which is known as the receptive field. The receptive field overlaps with the visual field, covering it entirely.

- CNNs perform better with image, speech, or audio inputs than other neural networks. There are three main layers in CNNs:
 - Convolutional layer
  - Pooling layer
 - Fully connected (FC) layer


## Overview: Neural Networks for Text or Time Series
The use of neural networks for time series forecasting has been widely adopted. Most often, these are feedforward networks that use a sliding window over the input sequence. Examples include market predictions, meteorological forecasting, and network traffic forecasting. Two of the most popular models used for time series are convolutional neural networks (CNNs) and long short-term memory networks (LSTMs).

Convolutional Neural Networks (CNNs)

CNNs can be used to predict time series from raw input data by learning and automatically extracting features. For example, an observation sequence can be treated as a one-dimensional image from which a CNN model can interpret and extract the salient elements.

Long Short-Term Memory Networks (LSTMs)
LSTMs, which add explicit order handling to the learning of a mapping function from inputs to outputs, are unavailable with MLPs or CNNs. They are neural networks that support sequences of observations as input data.

## Overview: Neural Networks for Regression and Reinforcement Learning and Its Application to Neural Networks
- In this section, you will learn about neural networks, in which simple units known as neurons are used as inputs and outputs (inspired by brain neurons).
- Furthermore, you will discover that neural networks are versatile and are used for both classification and regression. 
- You will also examine how neural networks can be applied to regression problems. Although neural networks are complex and computationally expensive, they are flexible. 
- They can dynamically choose the best type of regression, and you can add hidden layers to improve prediction further.

## Reinforcement Learning and Its Application to Neural Networks
Reinforcement learning (RL) is a type of ML where an agent learns to make decisions by interacting with an environment to maximize cumulative reward. The basics of reinforcement learning are:

Agent, Environment, and Actions
Agent: The learner or decision-maker that interacts with the environment. It makes decisions based on observations from the environment and aims to maximize rewards.
Environment: The external system with which the agent interacts. It provides feedback (rewards) and changes its state based on the agent’s actions.
Actions: Choices made by the agent that affect the environment. The set of all possible actions is called the action space.
State
A representation of the current situation or context in the environment. States define what the agent observes and helps determine the appropriate action.

Rewards
A scalar feedback signal received from the environment after the agent performs an action. Rewards can be positive (rewards) or negative (penalties) and are used to evaluate the action’s effectiveness.

Policy
 A strategy used by the agent to decide which action to take in a given state. It can be deterministic (always chooses the same action for a state) or stochastic (chooses actions probabilistically).

Value Function
A function that estimates the expected cumulative reward an agent can obtain starting from a state (state value) or state–action pair (action value). This helps the agent evaluate the long-term benefit of states or actions.

State Value Function (V(s)): Estimates the expected reward for being in state s and following a particular policy.
Action Value Function (Q(s, a)): Estimates the expected reward for taking action a in state s and following a particular policy thereafter.
Return
The total accumulated reward the agent receives over time, often discounted to prioritize immediate rewards. It can be calculated as a sum of rewards over episodes or steps.

Exploration vs. Exploitation
Exploration: The agent tries new actions to discover their effects and learn more about the environment.
Exploitation: The agent uses known information to select actions that maximize reward based on previous experience.
Temporal Difference (TD) Learning
 A method where the agent learns value functions based on the difference between successive predictions. TD learning updates estimates based on the difference between the predicted and observed rewards.

Module-Free vs. Model-Based Methods
Model-Free Methods: The agent learns to make decisions based solely on experience without using a model of the environment. Examples include Q-learning and SARSA.
Model-Based Methods: The agent uses a model of the environment to simulate outcomes and plan actions. These methods can be more efficient but require a model of the environment dynamics.
Algorithms
Reinforcement learning is widely used in various applications, such as robotics, game playing, autonomous vehicles, and finance, due to its ability to handle complex decision-making problems where traditional methods may fall short.

Recurrent neural networks (RNNs) are a specialized type of neural network designed to handle sequential data, which sets them apart from traditional neural networks.

Differences between RNNs and NNs
RNN	NN
Temporal Dynamics	Processes sequential data by maintaining an internal state that evolves over time, allowing the network to remember information from previous time steps and use it to influence future predictions.	Processes fixed-size input data and produce output without considering the order or sequence of the data. Lacks temporal dynamics.
Memory	Has a form of memory through hidden states that capture information about previous inputs in the sequence. This memory allows RNNs to make predictions based on the entire sequence of data.	Lacks explicit memory of past inputs.
Training	Training is more complex due to the need to handle the temporal dependencies and potential vanishing or exploding gradient problems.	Training involves updating weights based on the error between the predicted and actual outputs



