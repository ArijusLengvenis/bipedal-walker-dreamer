# Dreamwalker: Sample Efficient Soft Actor-Critic Agent with Dreamer Model

Dreamwalker is an advanced implementation of a Soft Actor-Critic model with Truncated Quantile Critics as well as Deep Dense Reinforcement learning, incorporating a Dreamer model to improve sample efficiency. The project focuses on a walking environment, specifically the [bipedal walker](https://www.gymlibrary.dev/environments/box2d/bipedal_walker/) and [bipedal walker hardcore](https://www.gymlibrary.dev/environments/box2d/bipedal_walker/) environments. The hyperparameters were optimised with [Optuna](https://optuna.org/) running on a GTX 1070 graphics card.

<div align="center">
    <img src="https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/gif/agent-video%2Cepisode%3D125%2Cscore%3D323.gif" alt="Running boy basic gif"/>
</div>

*BipedalWalker-v3 environment after 125 episodes of training. Achieved a score of 323 and converged on the policy.*

<div align="center">
    <img src="https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/gif/agent-hardcore-video%2Cepisode%3D750%2Cscore%3D313.gif" alt="Running boy advanced gif"/>
</div>

*BipedalWalkerHardcore-v3 environment after 750 episodes of training. Achieved a score of 313.*

## Technologies

The technologies used in this project include:

1. **Truncated Quantile Critics (TQC)** is a policy optimization method designed for continuous domains, aiming to reduce overestimation bias in Q-function approximation. Overestimation bias is a common issue in reinforcement learning, caused by over-optimistic approximations of the Q-function. It can be expressed mathematically by Jensen's inequality:

$$E [\max{Q(a) + U(a)}] \geq \max E[Q(a) + U(a)] = \max Q(a)$$

This essentially states that the maximum of the Q-function is not greater than the expected maximum of the approximate Q-function. Overestimation can be due to various factors including approximation errors, optimization imprecisions, and environmental uncertainties. TQC employs three key strategies to tackle this: distributional representation of a critic, truncation of the approximated distribution, and ensembling.

2. **Deep Dense Reinforcement Learning (D2RL)** enhances traditional Multi-layer Perceptron (MLP) architectures by using dense connections to improve the policy and value functions. In D2RL, the state or state-action pair is concatenated to each hidden layer, except for the last output linear layer. This straightforward architectural change enables an increase in network depth without major performance degradation. This is attributed to the Data Processing Inequality, which states that mutual information (MI) cannot be increased through local operations:

$$MI(X_1;X_2) \geq MI(X_1;X_3)$$

Essentially, as information moves through layers of a network, it cannot increase; it can only decrease or remain the same.

3. **Emphasizing Recent Experience (ERE) Buffer** is a replay buffer that gives priority to recent memories, thereby enhancing learning efficiency. As more recent memories are more likely to contain high-quality samples, the buffer gradually reduces the range of samples used for model updates, focusing more on recent data points and less on older ones. This process is mathematically represented by:

$$c_k = \max(N\eta^{\frac{1000k}{K}}, c_{min})$$

The value of η is slowly annealed (reduced) during the training process, gradually decreasing to 1, which represents uniform sampling:

$$\eta_t = \eta_0 + (\eta_T - \eta_0) \frac{t}{T}$$

4. **Dreamer Model**: A model-based approach using an auto-regressive transformer. The Dreamer model learns to simulate the environment, which will allow the agent to learn from it, increasing sample efficiency. The idea is to utilise an auto-regressive transformer trained on the memories of the agent to try to learn the environment conditioned on the previous states:

$$c(s^{t+1}, r^{t+1}, d^{t+1}| s^{t-n:t}, a^{t-n:t}, r^{t-n:t}, d^{t-n:t})$$

Given sufficient training data and epochs, this model can essentially replace the real environment, which would then only be necessary for validation of the agent.

## Repository Structure

This repository consists of a Jupyter notebook, `agent_code.ipynb`, which contains all the code related to methods, training, and testing loop of the Dreamwalker model. It also contains videos and logs of the attempt with seed 42 for everything that I was able to achieve with this code.

## Usage

To run the Dreamwalker model, follow these steps:

1. Clone the repository to your local machine.
2. Ensure that all dependencies are installed.
3. Open `agent_code.ipynb` in a Jupyter notebook environment.
4. Run all cells in the notebook.

## Results

The Dreamwalker agent outperforms vanilla TQC + D2RL implementation in terms of policy performance and sample efficiency. The agent successfully completes the Bipedal Walker environment and its hardcore variant in 74 and 256 episodes respectively. In the regular environment, the agent converges on the optimal policy after about 125 steps. In the hardcore environment, the agent reaches an average reward of 200 at episode 620 learning from scratch, reaching a maximum reward of 297 over the past 10 episodes after 1500 episodes.

<div align="center">
    <img src="https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/img/compare.png?raw=true" alt="compare graph"/>
    <img src="https://github.com/ArijusLengvenis/bipedal-walker-dreamer/blob/main/img/hardcore1500.png?raw=true" alt="hardcore graph"/>
</div>

*Fig. 1: Ablation analysis of Dreamwalker compared to more basic models. Fig. 2: the average reward of Dreamwalker of the last 10 steps in the hardcore environment.*


## Limitations and Future Work

Although the Dreamwalker agent shows promising results, there are some limitations. The current dreamer implementation utilises too simplistic architecture and loss functions, which may need to be revised to better simulate the environment and reduce the overestimation bias. The model's training time is also quite extensive due to the complexity of the Dreamer model and the large context length that could be required. Additionally, other replay buffers like the Priority Replay Experience (PRE) could be experimented with to see if it decreases convergence time.

## Original Sources

- **TQC:** Arsenii Kuznetsov et al. “Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics”. In: arXiv, 2020. doi: 10.48550/ARXIV.2005.04269. url: https://arxiv.org/abs/2005.0426
- **D2RL:** Samarth Sinha et al. “D2RL: Deep Dense Architectures in Reinforcement Learning”. In: arXiv, 2020. doi: 10.48550/ARXIV.2010.09163. url: https://arxiv.org/abs/2010.09163
- **ERE:** Che Wang and Keith Ross. “Boosting Soft Actor-Critic: Emphasizing Recent Experience without Forgetting the Past”. In: arXiv, 2019. doi: 10.48550/ARXIV.1906.04009. url: https://arxiv.org/abs/1906.040
- **Transformer Dreamer:** Chang Chen et al. “TransDreamer: Reinforcement Learning with Transformer World Models”. In: arXiv, 2022. doi: 10.48550/ARXIV.2202.09481. url: https://arxiv.org/abs/2202.0948

