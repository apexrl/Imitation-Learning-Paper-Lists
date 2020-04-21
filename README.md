# Imitation-Learning-Paper-Lists
Paper Collection for Imitation Learning in RL with brief introductions. This collection refers to [Awesome-Imitation-Learning](https://github.com/kristery/Awesome-Imitation-Learning) and also contains self-collected papers.

To be precise, the "imitation learning" is the general problem of learning from expert demonstration (LfD). There are 2 names derived from such a description, which are Imitation Learning and Apprenticeship Learning due to historical reasons. Usually, apprenticeship learning is mentioned in the context of "Apprenticeship learning via inverse reinforcement learning (IRL)" which recovers the reward function and learns policies from it, while imitation learning began with behavior cloning that learn the policy directly ([ref](https://cs.stackexchange.com/questions/56577/apprenticeship-vs-imitation-learning-what-is-the-difference) and <An autonomous land vehicle in a neural network> by Morgan-Kaufmann, NIPS 1989.). However, with the development of related researches, "imitation learning" is always used to represent the general LfD problem setting, which is also our view of point.

Typically, different settings of imitation learning derive to different specific areas. A general setting is that one can only obtain (1) pre-collected trajectories ((s,a) pairs) from uninteractive expert (2) he can interact with the environments (with simulators) (3) without reward signals. Here we list some of the other settings as below:

1. No actions and only state / observations -> Imitation Learning From Observations (ILFO).

2. With reward signals -> Imitation Learning with Rewards. 

3. Interactive expert for correctness and data aggregation -> On-policy Imitation Learning (begin as Dagger, Dataset Aggregation).

4. Can not interact with Environments -> A special case of Batch RL (see a particular list in [here](https://github.com/apexrl/Batch-RL-Paper-Lists), data in Batch RL can contain more than expert demos.)

What we want from imitation learning in different settings (for real world):

0. Less interact with the **real world** environments with expert demonstrations to improve sample efficiency and learn good policies. (yet some works use few demonstrations to learn good policies but with a vast cost on interacting with environments)

1. Real world actions are not available or hard to sample.

2. Use expert data to improve sample efficiency and learn fast with good exploration ability.

3. Some online setting that human are easily to join in, e.g., human correct the steering wheel in auto-driving cars.

4. Learn good policies in real world where interact with the environment is difficult.

In this collection, we will concentrate on the general setting and we collect other settings in "[Other Settings](https://github.com/apexrl/Imitation-Learning-Paper-Lists/blob/master/README.md#other-settings)" section. For other settings, such as "Self-imitation learning" which imitate the policy from one's own historical data, we do not regard it as an imitation learning task.

These papers are classified mainly based on their methodology instead and their specific task settings (except single-agent/multi-agent settings) but since there are many cross-domain papers, the classification is just for reference. As you can see, many works focus on Robotics, especially papers of UCB.

# Overview
* [Single-Agent](https://github.com/apexrl/Imitation-Learning-Paper-Lists#single-agent)
  * [Reveiws&Tutorials](https://github.com/apexrl/Imitation-Learning-Paper-Lists#reveiwstutorials)
  * [Behavior Cloning](https://github.com/apexrl/Imitation-Learning-Paper-Lists#behavior-cloning)
    * [One-shot / Zero-shot](https://github.com/apexrl/Imitation-Learning-Paper-Lists#one-shot--zero-shot)
    * [Model based](https://github.com/apexrl/Imitation-Learning-Paper-Lists#model-based)
    * [Hierarchical RL](https://github.com/apexrl/Imitation-Learning-Paper-Lists#hierarchical-rl)
    * [Multi-modal Behaviors](https://github.com/apexrl/Imitation-Learning-Paper-Lists#multi-modal-behaviors)
    * [Learning with human preference](https://github.com/apexrl/Imitation-Learning-Paper-Lists#learning-with-human-preference)
  * [Inverse RL](https://github.com/apexrl/Imitation-Learning-Paper-Lists#inverse-rl)
    * [Reveiws&Tutorials](https://github.com/apexrl/Imitation-Learning-Paper-Lists#reveiwstutorials-1)
    * [Papers](https://github.com/apexrl/Imitation-Learning-Paper-Lists#papers)
    * [Beyesian Methods](https://github.com/apexrl/Imitation-Learning-Paper-Lists#beyesian-methods)
  * [Generative Adversarial Methods](https://github.com/apexrl/Imitation-Learning-Paper-Lists#generative-adversarial-methods)
    * [Multi-modal Behaviors](https://github.com/apexrl/Imitation-Learning-Paper-Lists#multi-modal-behaviors-1)
    * [Hierarchical RL](https://github.com/apexrl/Imitation-Learning-Paper-Lists#hierarchical-rl-1)  
    * [Task Transfer](https://github.com/apexrl/Imitation-Learning-Paper-Lists#task-transfer)
    * [Model based](https://github.com/apexrl/Imitation-Learning-Paper-Lists#model-based-1)
    * [POMDP](https://github.com/apexrl/Imitation-Learning-Paper-Lists#pomdp)
    * [Beyesian Methods](https://github.com/apexrl/Imitation-Learning-Paper-Lists#beyesian-methods-1)
  * [Fixed Reward Methods](https://github.com/apexrl/Imitation-Learning-Paper-Lists#fixed-reward-methods)
  * [Goal-based methods](https://github.com/apexrl/Imitation-Learning-Paper-Lists#goal-based-methods)
  * [Other Methods](https://github.com/apexrl/Imitation-Learning-Paper-Lists#other-methods)
* [Multi-Agent](https://github.com/apexrl/Imitation-Learning-Paper-Lists#multi-agent)
  * [MA Inverse RL](https://github.com/apexrl/Imitation-Learning-Paper-Lists#ma-inverse-rl)
  * [MA-GAIL](https://github.com/apexrl/Imitation-Learning-Paper-Lists#ma-gail)
* [Other Settings](https://github.com/apexrl/Imitation-Learning-Paper-Lists#other-settings)
  * [Imitation Learning from Observations](https://github.com/apexrl/Imitation-Learning-Paper-Lists#imitation-learning-from-observations)
  * [Imitation Learning with Rewards](https://github.com/apexrl/Imitation-Learning-Paper-Lists#imitation-learning-with-rewards)
  * [On-policy Imitation Learning](https://github.com/apexrl/Imitation-Learning-Paper-Lists#on-policy-imitation-learning)
  * [Batch RL](https://github.com/apexrl/Imitation-Learning-Paper-Lists#batch-rl)
* [Applications](https://github.com/apexrl/Imitation-Learning-Paper-Lists#applications)

# Single-Agent

## Reveiws&Tutorials

* <[Imitation Learning Tutorial](https://sites.google.com/view/icml2018-imitation-learning/)> by Yisong Yue Hoang M. Le, ICML, 2018. (Video; Slide)

* <[Global overview of Imitation Learning](https://arxiv.org/abs/1801.06503)> by Alexandre Attia, Sharone Dayan, 2018.

* <[An Algorithmic Perspective on Imitation Learning](https://www.nowpublishers.com/article/Details/ROB-053)> by Takayuki Osa et al., 2018.

* <[Imitation learning: A survey of learning methods](https://dl.acm.org/citation.cfm?id=3071073.3054912)> by	Ahmed Hussein, Mohamed Medhat Gaber, Eyad Elyan,	Chrisina Jayne, 2017.

* <[Imitation learning basic Lecture (National Taiwan University)](https://www.youtube.com/watch?v=rOho-2oJFeA)> by Hongyi Li, 2017. (Video)

* <[New Frontiers in Imitation Learning](https://www.youtube.com/watch?v=4PnNlvPGbUQi)> by Yisong Yue, 2017. (Video)

* <[A survey of robot learning from demonstration](https://www.sciencedirect.com/science/article/pii/S0921889008001772)> by 
Brenna, D.Argall, SoniaChernova, ManuelaVeloso, BrettBrowning, 2009.

## Behavior Cloning

Behavior Cloning (BC) directly replicating the expert’s behavior with supervised learning, which can be improved via data aggregation. One can say that BC is the simplest case of interactive direct policy learning.

* <[Graph-Structured Visual Imitation](https://arxiv.org/abs/1907.05518)>, by Maximilian Sieb, Zhou Xian, Audrey Huang, Oliver Kroemer and Katerina Fragkiadaki, CoRL 2019.
  * This paper cast visual imitation as a visual correspondence problem in robotics. 
  
* <[Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979)> by Pim de Haan, Dinesh Jayaraman, Sergey Levine, 2019.

* [MetaMimic] <[One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL](https://arxiv.org/abs/1810.05017)>, Le Paine et al, 2018.
  * Propose **MetaMimic**.

* [DeepMimic] <[DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf)>, Peng et al, 2018. 
  * Propose **DeepMimic**.

* <[Hierarchical Imitation and Reinforcement Learning](https://arxiv.org/abs/1803.00590)> by Hoang M. Le, Nan Jiang, Alekh Agarwal, Miroslav Dudík, Yisong Yue and Hal Daumé III, 2018.

* <[Associate Latent Encodings in Learning from Demonstrations](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14509)> by Hang Yin, Francisco S. Melo, Aude Billard, Ana Paiva, 2017.

### One-shot / Zero-shot

* <[Zero-shot Imitation Learning from Demonstrations for Legged Robot Visual Navigation](https://arxiv.org/abs/1909.12971)> by Xinlei Pan, Tingnan Zhang, Brian Ichter, Aleksandra Faust, Jie Tan and Sehoon Ha, 2019.

* <[Zero-Shot Visual Imitation](https://arxiv.org/abs/1804.08606)>, by Deepak Pathak, Parsa Mahmoudieh, Guanghao Luo, Pulkit Agrawal, Dian Chen, Yide Shentu, Evan Shelhamer, Jitendra Malik, Alexei A. Efros and Trevor Darrell, ICLR 2018.

* <[One-Shot Hierarchical Imitation Learning of Compound Visuomotor Tasks](https://arxiv.org/pdf/1810.11043.pdf)>, by Tianhe Yu, Pieter Abbeel, Sergey Levine, Chelsea Finn et al., 2018.

* <[One-Shot Imitation Learning](https://arxiv.org/abs/1703.07326)>, by Yan Duan, Marcin Andrychowicz, Bradly C. Stadie, Jonathan Ho, Jonas Schneider, Ilya Sutskever, Pieter Abbeel and Wojciech Zaremba, NIPS 2017.

### Model based

* <[Safe end-to-end imitation learning for model predictive control](https://arxiv.org/abs/1803.10231)>, by Keuntaek Lee, Kamil Saigol and Evangelos A. Theodorou, ICRA 2019
  
* <[Deep Imitative Models for Flexible Inference, Planning, and Control](https://arxiv.org/abs/1810.06544)>, by Nicholas Rhinehart, Rowan McAllister and Sergey Levine, 2018. [[blog]](https://sites.google.com/view/imitative-models)

* <[Model-based imitation learning from state trajectories](https://openreview.net/forum?id=S1GDXzb0b&noteId=S1GDXzb0b)>, by S. Chaudhury et al., 2018.

### Hierarchical RL

* <[Hierarchical Imitation and Reinforcement Learning](https://arxiv.org/abs/1803.00590)>, by Hoang M. Le, Nan Jiang, Alekh Agarwal, Miroslav Dudík, Yisong Yue and Hal Daumé III, ICML 2018

### Multi-modal Behaviors

* <[Watch, Try, Learn: Meta-Learning from Demonstrations and Reward. Imitation learning](https://arxiv.org/pdf/1906.03352)>, by Allan Zhou, Eric Jang, Daniel Kappler, Alex Herzog, Mohi Khansari, Paul Wohlhart, Yunfei Bai, Mrinal Kalakrishnan, Sergey Levine and Chelsea Finn, 2019.

* <[Learning a Multi-Modal Policy via Imitating Demonstrations with Mixed Behaviors](https://arxiv.org/pdf/1903.10304)>, by Fang-I Hsiao, Jui-Hsuan Kuo, Min Sun, NIPS 2018 Workshop.

* <[Shared Multi-Task Imitation Learning for Indoor Self-Navigation](https://arxiv.org/abs/1808.04503)>, by Junhong Xu, Qiwei Liu, Hanqing Guo, Aaron Kageza, Saeed AlQarni and Shaoen Wu, 2018

### Learning with human preference
* <[A Low-Cost Ethics Shaping Approach for Designing Reinforcement Learning Agents](https://arxiv.org/abs/1712.04172)>, by Yueh-Hua Wu, Shou-De Lin, AAAI 2018

* <[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)>, by Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg and Dario Amodei, NIPS 2017.

## Inverse RL

Inverse Rinforcement Learning (IRL) learns hidden objectives of the expert’s behavior.

### Reveiws&Tutorials

* <[Inverse Reinforcement Learning](https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html#algorithms)> by Johannes Heidecke, 2018. (Blog)

* <[Inverse Reinforcement Learning](https://towardsdatascience.com/inverse-reinforcement-learning-6453b7cdc90d)> by Alexandre Gonfalonieri, 2018.（Blog）

* <[A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress](https://arxiv.org/abs/1806.06877)> by Saurabh Arora, Prashant Doshi, 2012.

* <[A survey of inverse reinforcement learning techniques](https://www.emerald.com/insight/content/doi/10.1108/17563781211255862/full/html)> by Shao Zhifei, Er Meng Joo, 2012.

* <[A review of inverse reinforcement learning theory and recent advances](https://ieeexplore.ieee.org/abstract/document/6256507)> by Shao Zhifei, Er Meng Joo, 2012.

* <[From Structured Prediction to Inverse Reinforcement Learning](https://www.aclweb.org/anthology/P10-5005/) by Hal Daumé III, 2010.

### Papers

* <[Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/abs/1810.00821)>, by Xue Bin Peng, Angjoo Kanazawa, Sam Toyer, Pieter Abbeel and Sergey Levine, 2018.
  * Propose **VAIL**.

* <[Learning to Optimize via Wasserstein Deep Inverse Optimal Control
](https://arxiv.org/abs/1805.08395)> by Yichen Wang, Le Song, and Hongyuan Zha, 2018.

* [AIRL] <([Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/pdf/1710.11248)>, by Justin Fu, Katie Luo, Sergey Levine, ICLR 2018.

* <[A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models
](https://arxiv.org/abs/1611.03852)> by Chelsea Finn, Paul Christiano, Pieter Abbeel, Sergey Levine, 2016.

* [GCL] <[Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448)>, by Chelsea Finn, Sergey Levine, Pieter Abbeel, 2016. 
  * Propose **GCL.**

* <[Infinite Time Horizon Maximum Causal Entropy Inverse Reinforcement Learning
 ](https://ieeexplore.ieee.org/document/7040156)> by Michael Bloem and Nicholas Bambos, 2014.

* <[Maximum Likelihood Inverse Reinforcement Learning](http://cs.brown.edu/~mlittman/theses/babes.pdf)> by MONICA C. VROMAN, 2014.

* <[The Principle of Maximum Causal Entropy
for Estimating Interacting Processes](http://ieeexplore.ieee.org/abstract/document/6479340/)> by Brian D. Ziebart, J. Andrew Bagnell, and Anind K. Dey, 2012.

* <[Apprenticeship Learning using Inverse Reinforcement Learning and Gradient Methods](https://arxiv.org/abs/1206.5264)> by Gergely Neu, Csaba Szepesvari, 2012.

* <[Nonlinear Inverse Reinforcement Learning with Gaussian Processes](http://papers.nips.cc/paper/4420-nonlinear-inverse-reinforcement-learning-with-gaussian-processes)> by Sergey Levine, Zoran Popovic and Vladlen Koltun, 2011.

* <[Relative Entropy Inverse Reinforcement Learning](http://www.jmlr.org/proceedings/papers/v15/boularias11a/boularias11a.pdf)> by Abdeslam Boularias, Jens Kober and Jan Peters, 2011.

* <[Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy](http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)>, Ziebart 2010.
  * **Contributions:** Crisp formulation of maximum entropy IRL.

* <[Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)> by 
Brian D. Ziebart, Andrew Maas, J.Andrew Bagnell, and Anind K. Dey, 2008.

* <[Apprenticeship Learning Using Linear Programming](http://rob.schapire.net/papers/SyedBowlingSchapireICML2008.pdf)> by Umar Syed, Michael Bowling and Robert E. Schapire, 2008.

* <[Bayesian Inverse Reinforcement Learning](https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-416.pdf)> by Deepak Ramachandran and Eyal Amir, 2007.

* <[Apprenticeship learning via inverse reinforcement learning](https://dl.acm.org/citation.cfm?id=1015430)> by	Pieter Abbeel, Andrew Y. Ng, ICML 2004.
    * Learn a policy reaches the reward that is recovered from demonstrations, which is not worse $\epsilon$ than the optimal (expert) policy under the real reward function.

* <[Algorithms for Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)> by AY Ng, SJ Russell, 2000.

### Beyesian Methods

* <[Bayesian Inverse Reinforcement Learning]>(https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-416.pdf)
 by D Ramachandran, E Amir, IJCAI 2007. 

## Generative Adversarial Methods

Generative Adversarial Imitation Learning (GAIL) apply generative adversarial training manner into learning expert policies, which is derived from inverse RL.

* <[Adversarial Imitation Learning from Incomplete Demonstrations](https://arXiv.org/abs/1905.12310)> by Mingfei Sun and Xiaojuan Ma, 2019.

* [WGAN-GAIL] <[Wasserstein Adversarial Imitation Learning](https://arxiv.org/abs/1906.08113)>, by Huang Xiao, Michael Herman, Joerg Wagner, Sebastian Ziesche, Jalal Etesami and Thai Hong Linh, 2019

* <[Self-Improving Generative Adversarial Reinforcement Learning
 ](https://dl.acm.org/citation.cfm?id=3331673)> by Yang Liu, Yifeng Zeng, Yingke Chen, Jing Tang, Yinghui Pan, 2019.

* [GMMIL] <[Imitation Learning via Kernel Mean Embedding](https://www-users.cs.umn.edu/~hspark/mmd.pdf)> by Kee-Eung Kim, Hyun Soo Park, AAAI 2018.
 
* [Off-Policy GAIL] <[Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning
](https://arxiv.org/abs/1809.02925)> by Ilya Kostrikov, Kumar Krishna Agrawal, Debidatta Dwibedi, Sergey Levine, Jonathan Tompson, 2018.

* [RAIL] <[RAIL: Risk-Averse Imitation Learning](https://arxiv.org/abs/1707.06658)>, by Anirban Santara, Abhishek Naik, Balaraman Ravindran, Dipankar Das, Dheevatsa Mudigere, Sasikanth Avancha, Bharat Kaul, NIPS 2017.

* [GAIL] <[Generative Adversarial Imitation Learning](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning)> by Jonathan Ho and Stefano Ermon, 2016.
  * Propose **GAIL**, minimize the JS divergence of policy and expert policy with GAN's technique.
  
### Multi-modal Behaviors

* <[Learning Plannable Representations with Causal InfoGAN](http://papers.nips.cc/paper/8090-learning-plannable-representations-with-causal-infogan.pdf)>, by Thanard Kurutach, Aviv Tamar, Ge Yang, Stuart Russelland Pieter Abbeel, NeurIPS 2018.

* [InfoGAIL] <[InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations](http://papers.nips.cc/paper/6971-infogail-interpretable-imitation-learning-from-visual-demonstrations)> by Yunzhu Li, Jiaming Song and Stefano Ermon, 2017.
  * Propose **InfoGAIL**, learn interpretable (diverse) policies with mutual information technique (See InfoGAN).
  
* <[Multi-Modal Imitation Learning from Unstructured Demonstrations using Generative Adversarial Nets](https://arxiv.org/abs/1705.10479)>, by Karol Hausman, Yevgen Chebotar, Stefan Schaal, Gaurav Sukhatme, Joseph Lim., NIPS 2017.

* <[Robust Imitation of Diverse Behaviors
](https://arxiv.org/abs/1707.02747)> by Arjun Sharma, Mohit Sharma, Nicholas Rhinehart, Kris M. Kitani, NIPS 2017.

* <[Multi-task policy search for robotics](https://ieeexplore.ieee.org/abstract/document/6907421)> by Marc Peter Deisenroth, Peter Englert, Jan Peters, Dieter Fox, ICRA 2014.
  
### Hierarchical RL

* <[Directed-Info GAIL: Learning Hierarchical Policies from Unsegmented Demonstrations using Directed Information](https://arxiv.org/abs/1810.01266)> by Arjun Sharma, Mohit Sharma, Nicholas Rhinehart, Kris M. Kitani, ICLR 2019.

* [OptionGAN: Learning Joint Reward-Policy Options using Generative Adversarial Inverse Reinforcement Learning](https://arxiv.org/pdf/1709.06683.pdf), by Peter Henderson, Wei-Di Chang, Pierre-Luc Bacon, David Meger, Joelle Pineau and Doina Precup, AAAI 2018.

### Task Transfer

* <[Cross Domain Imitation Learning](https://arxiv.org/abs/1910.00105)> by Kun Ho Kim, Yihong Gu, Jiaming Song, Shengjia Zhao, Stefano Ermon, 2019.

* [TRAIL] <[Task-Relevant Adversarial Imitation Learning](https://arxiv.org/abs/1910.01077)>, by Konrad Zolna, Scott Reed, Alexander Novikov, Sergio Gomez Colmenarej, David Budden, Serkan Cabi, Misha Denil, Nando de Freitas, Ziyu Wang, 2019.

### Model-based

* <[Model Imitation for Model-Based Reinforcement Learning](https://arxiv.org/pdf/1909.11821.pdf)> by Yueh-Hua Wu, Ting-Han Fan, Peter J. Ramadge, Hao Su, 2019.

* [Planning, Dyna-AIL] <[Dyna-AIL : Adversarial Imitation Learning by Planning](https://arxiv.org/abs/1903.03234)>, by Vaibhav Saxena, Srinivasan Sivanandan, Pulkit Mathur, 2019.

* <[End-to-End Differentiable Adversarial Imitation Learning
 ](http://proceedings.mlr.press/v70/baram17a.html)> by Nir Baram, Oron Anschel, Itai Caspi, Shie Mannor, 2017.

### POMDP

* <[Learning Belief Representations for Imitation Learning in POMDPs](https://arxiv.org/abs/1906.09510)> by Tanmay Gangwani, Joel Lehman, Qiang Liu, Jian Peng, 2019.
  
## Fixed Reward Methods

Recently, there is a paper designs a new idea for imitation learning, which learns a fixed reward signal which obviates the need for dynamic update of reward functions.

* <[Disagreement-Regularized Imitation Learning](https://openreview.net/forum?id=rkgbYyHtwB&noteId=Syx2DuzwFr)> by Anonymous, (Submitted to ICLR) 2019.

* <[Support-guided Adversarial Imitation Learning](https://openreview.net/forum?id=r1x3unVKPS)> by Anonymous, (Submitted to ICLR) 2019.

* [SQIL] <[SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards](https://arxiv.org/abs/1905.11108)> by Siddharth Reddy, Anca D. Dragan, Sergey Levine. (Submitted to ICLR) 2019.

* <[Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation](https://arxiv.org/pdf/1905.06750)> by Ruohan Wang, Carlo Ciliberto, Pierluigi Amadori and Yiannis Demirisn, 2019.
  * Propose **RED**, use RND for IL.
  
## Goal-based methods

* [GoalGAIL] <[Goal-conditioned Imitation Learning](https://arxiv.org/abs/1906.05838)> by Yiming Ding, Carlos Florensa, Mariano Phielipp and Pieter Abbeel, ICML 2019.

* <[Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/abs/1906.05838)> by Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel, ICRA 2018.

### Beyesian Methods

* <[A Bayesian Approach to Generative Adversarial Imitation Learning]>(http://ailab.kaist.ac.kr/papers/jeon2018bayesian)
 by Wonseok Jeon, Seokin Seo, and Kee-Eung Kim, NIPS 2018. 
  
## Other Methods

* <[A Divergence Minimization Perspective on Imitation Learning Methods](http://arxiv.org/abs/1911.02256)> by Seyed Kamyar Seyed Ghasemipour, Richard Zemel and Shixiang Gu, 2019.

* <[Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732)>, by Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Gabriel Dulac-Arnold, Ian Osband, John Agapiou, Joel Z. Leibo, Audrunas Gruslys, AAAI 2018.

* <[Observe and look further: Achieving consistent performance on atari](https://arxiv.org/abs/1805.11593)> by Tobias Pohlen, Bilal Piot, Todd Hester, Mohammad Gheshlaghi Azar, Dan Horgan, David Budden, Gabriel Barth-Maron, Hado van Hasselt, John Quan, Mel Veˇcerík, et al, Arxiv 2018.

* <[Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)> by Justin Fu, Katie Luo, Sergey Levine, 2018.

* [UPN] <[Universal Planning Networks](https://arxiv.org/abs/1804.00645)>, by Aravind Srinivas, Allan Jabri, Pieter Abbeel, Sergey Levine, Chelsea Finn, 2018.

* <[Learning to Search via Retrospective Imitation](https://arxiv.org/abs/1804.00846)>, Jialin Song, Ravi Lanka, Albert Zhao, Aadyot Bhatnagar, Yisong Yue, Masahiro Ono, 2018.
    * This paper is for combinatorial problems.

* <[Third-Person Imitation Learning](https://arxiv.org/abs/1703.01703)>, by Bradly C. Stadie, Pieter Abbeel and Ilya Sutskever, ICLR 2017

* [Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Reward](https://pdfs.semanticscholar.org/8186/04245973bb30ad021728149a89157b3b2780.pdf), Mel Vecerik, Todd Hester, Jonathan Scholz, Fumin Wang, Olivier Pietquin, Bilal Piot, Nicolas Heess, Thomas Rothörl, Thomas Lampe and Martin Riedmiller, 2017.


* [Model-based Imitation Learning by Probabilistic Trajectory Matching](https://ieeexplore.ieee.org/abstract/document/6630832), Peter Englert ; Alexandros Paraschos ; Jan Peters ; Marc Peter Deisenroth, ICRA 2013. 


* [Imitation Learning Using Graphical Models](https://link.springer.com/chapter/10.1007/978-3-540-74958-5_77), Deepak VermaRajesh, P.N.Rao, ECML2007.

# Multi-Agent

* <[PRECOG: PREdiction Conditioned On Goals in Visual Multi-Agent Settings](https://arxiv.org/abs/1905.01296)> by Nicholas Rhinehart, Rowan McAllister, Kris Kitani and Sergey Levine, ICCV 2019. [[blog]](https://sites.google.com/view/precog)

* <[Imitation Learning of Factored Multi-agent Reactive Models](https://arxiv.org/abs/1903.04714) by Michael Teng, Tuan Anh Le, Adam Scibior, Frank Wood, 2019.

* <[Coordinated multi-agent imitation learning](https://dl.acm.org/citation.cfm?id=3305587)> by Dylan Hadfield-Menell, Stuart J. Russell, Pieter Abbeel and Anca Dragan, NIPS 2016.

## MA Inverse RL

* <[Cooperative Inverse Reinforcement Learning](http://papers.nips.cc/paper/6420-cooperative-inverse-reinforcement-learning)> by 	Hoang M. Le, Yisong Yue, Peter Carr, Patrick Lucey,	ICML 2017.

* <[Multi-Agent Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1907.13220)> by Lantao Yu, Jiaming Song, Stefano Ermon. ICML 2019.

* <[Comparison of Multi-agent and Single-agent Inverse Learning on a Simulated Soccer Example](https://arxiv.org/pdf/1403.6822.pdf)> by Lin X, Beling P A, Cogill R, arXiv 2014.

* <[Multi-agent inverse reinforcement learning for zero-sum games](https://arxiv.org/pdf/1403.6508.pdf)> by Lin X, Beling P A, Cogill R, arXiv 2014.

* <[Multi-robot inverse reinforcement learning under occlusion with interactions](http://aamas2014.lip6.fr/proceedings/aamas/p173.pdf)> by Bogert K, Doshi P, AAMAS 2014.

* <[Multi-agent inverse reinforcement learning](http://homes.soic.indiana.edu/natarasr/Papers/mairl.pdf)> by Natarajan S, Kunapuli G, Judah K, et al, ICMLA 2010.

## MA-GAIL

* <[Multi-Agent Generative Adversarial Imitation Learning](https://papers.nips.cc/paper/7975-multi-agent-generative-adversarial-imitation-learning)> by Jiaming Song, Hongyu Ren, Dorsa Sadigh, Stefano Ermon, NeurIPS 2018.

# Other Settings

## Imitation Learning from Observations

### Review Papers

* <[Recent Advances in Imitation Learning from Observation](https://arxiv.org/pdf/1905.13566.pdf)>, by Faraz Torabi, Garrett Warnell, Peter Stone, IJCAI 2019.

### Regular Papers

* <[Imitation Learning from Observations by Minimizing Inverse Dynamics Disagreement](https://arxiv.org/abs/1910.04417)>, by Chao Yang, Xiaojian Ma, Wenbing Huang, Fuchun Sun, Huaping Liu, Junzhou Huang, Chuang Gan, NeurIPS 2019.

* [Provably Efficient Imitation Learning from Observation Alone](http://proceedings.mlr.press/v97/sun19b.html), by Wen Sun, Anirudh Vemula, Byron Boots and Drew Bagnell, ICML 2019.

* <[To Follow or not to Follow: Selective Imitation Learning from Observations](https://arxiv.org/abs/1707.03374)>, by YuXuan Liu, Abhishek Gupta, Pieter Abbeel and Sergey Levine, CoRL 2019.

* <[Adversarial Imitation Learning from State-only Demonstrations](https://dl.acm.org/citation.cfm?id=3332067)>, by	Faraz Torabi, Garrett Warnell and Peter Stone, AAMAS 2019.

* <[Behavioral Cloning from Observation](https://arxiv.org/abs/1805.01954)>, by Faraz Torabi, Garrett Warnell and Peter Stone, IJCAI 2018.

* <[Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation](https://arxiv.org/abs/1707.03374)>, by YuXuan Liu, Abhishek Gupta, Pieter Abbeel and Sergey Levine, 2017.

* <[Observational Learning by Reinforcement Learning](https://arxiv.org/abs/1706.06617)>, by Diana Borsa, Bilal Piot, Rémi Munos and Olivier Pietquin, 2017.

## Imitation Learning with rewards

* <[Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning](https://arxiv.org/abs/1910.11956)>, by Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine and Karol Hausman, CoRL 2019

* <[Integration of Imitation Learning using GAIL and Reinforcement Learning using Task-achievement Rewards via Probabilistic Generative Model](https://arxiv.org/pdf/1907.02140.pdf)>, by Akira Kinose and Tadahiro Taniguchi, 2019.

* <[Reinforced Imitation in Heterogeneous Action Space](https://arxiv.org/pdf/1904.03438)>, by Konrad Zolna, Negar Rostamzadeh, Yoshua Bengio, Sungjin Ahn and Pedro O. Pinheiro, 2019.

* <[Policy Optimization with Demonstrations](http://proceedings.mlr.press/v80/kang18a.html)>, by Bingyi Kang, Zequn Jie and Jiashi Feng, ICML 2018.

* <[Reinforcement Learning from Imperfect Demonstrations](https://arxiv.org/pdf/1802.05313.pdf)>, by Yang Gao, Huazhe Xu, Ji Lin, Fisher Yu, Sergey Levine and Trevor Darrell, ICML Workshop 2018

* <[Pre-training with Non-expert Human Demonstration for Deep Reinforcement Learning](https://arxiv.org/pdf/1812.08904)>, by Gabriel V. de la Cruz, Yunshu Du and Matthew E. Taylor, 2018.

* <[Sparse Reward Based Manipulator Motion Planning by Using High Speed Learning from Demonstrations](https://ieeexplore.ieee.org/abstract/document/8665328)>, by Guoyu Zuo, Jiahao Lu and Tingting Pan, ROBIO 2018.

## On-policy Imitation Learning

* <[On-Policy Imitation Learning from an Improving Supervisor](https://realworld-sdm.github.io/paper/7.pdf)> by Ashwin Balakrishna and Brijen Thananjeyan, ICML Workshop 2019.

* <[On-Policy Robot Imitation Learning from a Converging Supervisor](https://arxiv.org/abs/1907.03423)>, Ashwin Balakrishna, Brijen Thananjeyan, Jonathan Lee, Felix Li, Arsh Zahed, Joseph E. Gonzalez and Ken Goldberg, CoRL 2019.
  * This paper consider "converging supervisor" in imitation learning as a dynamic DAgger methods.

* [DAgger] <[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)> by Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, 2011.

* [PoicyAggregation-SMILe] <[Efficient reductions for imitation learning](http://www.jmlr.org/proceedings/papers/v9/ross10a/ross10a.pdf)> by S Ross, D Bagnell, 2010.

* [PoicyAggregation-SEARN] <[Search-based Structured Prediction](https://arxiv.org/abs/0907.0786)> by Hal Daumé III, John Langford, Daniel Marcu, 2009.

## Batch RL

see a particular list in [here](https://github.com/apexrl/Batch-RL-Paper-Lists).

# Applications

* <[Better-than-Demonstrator Imitation Learning via Automatically-Ranked Demonstrations](https://arxiv.org/pdf/1907.03976.pdf)>, by Daniel S. Brown, Wonjoon Goo, Scott Niekum, CoRL 2019.

* <[Multi-Task Hierarchical Imitation Learning for Home Automation](http://ronberenstein.com/papers/CASE19_Multi-Task%20Hierarchical%20Imitation%20Learning%20for%20Home%20Automation%20%20.pdf)>, by Fox Roy, Berenstein Ron, Stoica Ion and Goldberg Ken, 2019.

* <[Imitation Learning for Human Pose Prediction](https://arxiv.org/pdf/1909.03449.pdf)>, by Borui Wang, Ehsan Adeli, Hsu-kuang Chiu, De-An Huang and Juan Carlos Niebles, 2019.

* <[Making Efficient Use of Demonstrations to Solve Hard Exploration Problems](https://arxiv.org/abs/1909.11821)>, by Tom Le Paine, Caglar Gulcehre, Bobak Shahriari, Misha Denil, Matt Hoffman, Hubert Soyer, Richard Tanburn, Steven Kapturowski, Neil Rabinowitz, Duncan Williams, Gabriel Barth-Maron, Ziyu Wang, Nando de Freitas and Worlds Team, 2019.

* <[Imitation Learning from Video by Leveraging Proprioception](https://arxiv.org/pdf/1905.09335.pdf)>, by Faraz Torabi, Garrett Warnell and Peter Stone, IJCAI 2019.

* <[Reinforcement and Imitation Learning for Diverse Visuomotor Skills](https://arxiv.org/abs/1802.09564)>, Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool, János Kramár, Raia Hadsell, Nando de Freitas and Nicolas Heess, RSS 2018.

* <[End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)>, by Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, Alexey Dosovitskiy, ICRA 2018.

* <[End-to-End Learning Driver Policy using Moments Deep Neural Network](https://ieeexplore.ieee.org/abstract/document/8664869)>, by Qian Deheng, Ren Dongchun, Meng Yingying, Zhu Yanliang, Ding Shuguang, Fu Sheng, Wang Zhichao and Xia Huaxia, ROBIO 2018.

* <[R2P2: A ReparameteRized Pushforward Policy for Diverse, Precise Generative Path Forecasting](https://link.springer.com/chapter/10.1007/978-3-030-01261-8_47)>, by Qian Deheng, Ren Dongchun, Meng Yingying, Zhu Yanliang, Ding Shuguang, Fu Sheng, Wang Zhichao and Xia Huaxia, ECCV 2018. [[blog]](http://www.cs.cmu.edu/~nrhineha/R2P2.html) 

* <[Learning Montezuma’s Revenge from a Single Demonstration](https://arxiv.org/pdf/1812.03381.pdf)>, by Tim Salimans and Richard Chen, 2018.
  
* <[ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/pdf/1812.03079.pdf)>, by Mayank Bansal, Alex Krizhevsky and Abhijit Ogale, 2018.
  
* <[Video Imitation GAN: Learning control policies by imitating raw videos using generative adversarial reward estimation](https://arxiv.org/pdf/1810.01108.pdf)>, by Subhajit Chaudhury, Daiki Kimura, Asim Munawar and Ryuki Tachibana, 2018.

* <[Query-Efficient Imitation Learning for End-to-End Autonomous Driving](https://arxiv.org/abs/1605.06450)>, by Jiakai Zhang and Kyunghyun Cho, 2016.

