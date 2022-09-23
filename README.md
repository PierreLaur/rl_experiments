# imitation_learning
Minimal implementations of Imitation Learning algorithms.  
Mainly done during my internship at 3IT Sherbrooke - Humanoid and Collaborative Robotics team  
Please feel free to email me if you have any questions.  
Pierre Laur [plaur@insa-toulouse.fr](mailto:plaur@insa-toulouse.fr)  

- Install requirements with ```pip install -r requirements.txt```
- Files that start with a number can be directly launched, the rest are for usage in other files

* Includes minimal implementations of RL algorithms. Tested with OpenAI Gym & neural nets implemented in Tensorflow
	- Recent RL algorithms :
		* [Soft Actor-Critic - Haarnoja et al. 2018](https://arxiv.org/abs/1812.05905)
		* [Proximal Policy Optimization - Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
		* [Deep Deterministic Policy Gradients - Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971)
		* [DQN - Mnih et al. 2013](https://arxiv.org/abs/1312.5602) & [Double DQN - Hasselt et al. 2015](https://arxiv.org/abs/1509.06461)

	- Older algorithms (for basic understanding of RL) : (implementations are based on [the Sutton&Barto RL textbook](http://incompleteideas.net/book/the-book-2nd.html))
		* REINFORCE
		* Actor-Critic
		* TD Learning : Q-Learning, SARSA, Expected SARSA, Dyna-Q, Double Q-Learning
		* TD Learning with Tile Coding (for continuous state spaces)
		* Q-Learning with Prioritized Sweeping
		* On-Policy Monte Carlo Control
		* Dynamic Programming (Policy Iteration & Value Iteration)

* Includes experiments with Random Search & Bayesian Optimization for hyperparameter optimization in RL (*hp_tuning* folder)
