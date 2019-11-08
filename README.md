# Imitation-Learning-Paper-Lists
Paper Collection for Imitation Learning in RL with brief introductions. This collection refer to [Awesome-Imitation-Learning](https://github.com/kristery/Awesome-Imitation-Learning) and also contains self-collected papers.

To be precise, the "imitation learning" is the general problem of learning from expert demonstration (LfD). There are 2 names derived from such a description, which are Imitation Learning and Apprenticeship Learning due to historical reasons. Usually, apprenticeship learning is mentioned in the context of "Apprenticeship learning via inverse reinforcement learning (IRL)" which recovers the reward function and learns policies from it, while imitation learning began with behavior cloning that learn the policy directly [ref](https://cs.stackexchange.com/questions/56577/apprenticeship-vs-imitation-learning-what-is-the-difference). However, with the development of related researches, "imitation learning" is always used to represent the general LfD problem setting, which is also our view of point.

Typically, methods collected in this collection do not assume to ask for an interactive expert for correctness and data aggregation, but when it is allowed, then lead to series of interactive direct policy learning methods, which is often analyzed via learning reductions. Since we do not expect for an interactive expert, we only list the original DAgger (Dataset Aggregation) paper and the original policy aggeration papers in [Behavior-Cloning](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Behavior-Cloning), and we will concentrate on those who only learn from pre-collected demonstrations.

## Overview
* [Single-Agent](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Single-Agent)
  * [Reveiws&Tutorials](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Reveiws\&Tutorials)
  * [Behavior-Cloning](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Behavior-Cloning)
  * [Inverse-RL](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Inverse-RL)
  * [Generative-Adversarial-Methods](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Generative-Adversarial-Methods)
  * [Support Estimation](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Support-Estimation)
  * 
* [Multi-Agent](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#Multi-Agent)
  * [MA-Inverse-RL](https://github.com/Ericonaldo/Imitation-Learning-Paper-Lists#MA-Inverse-RL)

## Single-Agent

## Reveiws&Tutorials

* <[Imitation Learning Tutorial](https://sites.google.com/view/icml2018-imitation-learning/)> by Yisong Yue Hoang M. Le, ICML, 2018. (Vedio; Slide)

* <[Global overview of Imitation Learning](https://arxiv.org/abs/1801.06503)> by Alexandre Attia, Sharone Dayan, 2018.

* <[An Algorithmic Perspective on Imitation Learning](https://www.nowpublishers.com/article/Details/ROB-053)> by Takayuki Osa et al., 2018.

* <[Imitation learning: A survey of learning methods](https://dl.acm.org/citation.cfm?id=3071073.3054912)> by	Ahmed Hussein, Mohamed Medhat Gaber, Eyad Elyan,	Chrisina Jayne, 2017.

* <[Imitation learning basic Lecture (National Taiwan University)](https://www.youtube.com/watch?v=rOho-2oJFeA)> by Hongyi Li, 2017. (Vedio)

* <[New Frontiers in Imitation Learning](https://www.youtube.com/watch?v=4PnNlvPGbUQi)> by Yisong Yue, 2017. (Vedio)

* <[A survey of robot learning from demonstration](https://www.sciencedirect.com/science/article/pii/S0921889008001772)> by 
Brenna, D.Argall, SoniaChernova, ManuelaVeloso, BrettBrowning, 2009.

## Behavior-Cloning

Behavior Cloning (BC) directly replicating the expert’s behavior with supervised learning, which can be improved via data aggregation. One can say that BC is the simplest case of interactive direct policy learning.

* <[Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979)> by Pim de Haan, Dinesh Jayaraman, Sergey Levine, 2019.

* <[One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL](https://arxiv.org/abs/1810.05017)>, Le Paine et al, 2018.
  * Propose **MetaMimic**.

* <[DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf)>, Peng et al, 2018. 
  * Propose **DeepMimic**.

* <[Hierarchical Imitation and Reinforcement Learning](https://arxiv.org/abs/1803.00590)> by Hoang M. Le, Nan Jiang, Alekh Agarwal, Miroslav Dudík, Yisong Yue, Hal Daumé III, 2018.

* <[Associate Latent Encodings in Learning from Demonstrations](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14509)> by Hang Yin, Francisco S. Melo, Aude Billard, Ana Paiva, 2017.

* [DAgger] <[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)> by Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, 2011.

* [PoicyAggregation-SMILe] <[Efficient reductions for imitation learning](http://www.jmlr.org/proceedings/papers/v9/ross10a/ross10a.pdf)> by S Ross, D Bagnell, 2010.

* [PoicyAggregation-SEARN] <[Search-based Structured Prediction](https://arxiv.org/abs/0907.0786)> by Hal Daumé III, John Langford, Daniel Marcu, 2009.

## Inverse-RL

Inverse Rinforcement Learning (IRL) learns hidden objectives of the expert’s behavior.

### Reveiws&Tutorials

* <[Inverse Reinforcement Learning](https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html#algorithms)> by Johannes Heidecke, 2018. (Blog)

* <[Inverse Reinforcement Learning](https://towardsdatascience.com/inverse-reinforcement-learning-6453b7cdc90d)> by Alexandre Gonfalonieri, 2018.（Blog）

* <[A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress](https://arxiv.org/abs/1806.06877)> by Saurabh Arora, Prashant Doshi, 2012.

* <[A survey of inverse reinforcement learning techniques](https://www.emerald.com/insight/content/doi/10.1108/17563781211255862/full/html)> by Shao Zhifei, Er Meng Joo, 2012.

* <[A review of inverse reinforcement learning theory and recent advances](https://ieeexplore.ieee.org/abstract/document/6256507)> by Shao Zhifei, Er Meng Joo, 2012.

* <[From Structured Prediction to Inverse Reinforcement Learning](https://www.aclweb.org/anthology/P10-5005/) by Hal Daumé III, 2010.

### Papers

* <[Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/abs/1810.00821)>, Peng et al, 2018.
  * Propose **VAIL**.

* <[Learning to Optimize via Wasserstein Deep Inverse Optimal Control
](https://arxiv.org/abs/1805.08395)> by Yichen Wang, Le Song, and Hongyuan Zha, 2018.

* <[Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)> by Justin Fu, Katie Luo, Sergey Levine, 2018.

* <[A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models
](https://arxiv.org/abs/1611.03852)> by Chelsea Finn, Paul Christiano, Pieter Abbeel, Sergey Levine, 2016.

* <[Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448)>, Finn et al, 2016. **Algorithm: GCL.**

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

## Generative-Adversarial-Methods

Generative Adversarial Imitation Learning (GAIL) apply generative adversarial training manner into learning expert policies, which is derived from inverse RL.

* [Model based] <[Model Imitation for Model-Based Reinforcement Learning](https://arxiv.org/pdf/1909.11821.pdf)> by Yueh-Hua Wu, Ting-Han Fan, Peter J. Ramadge, Hao Su, 2019.

* [Planing, Dyna-AIL] <[Dyna-AIL : Adversarial Imitation Learning by Planning](https://arxiv.org/abs/1903.03234)>, by Vaibhav Saxena, Srinivasan Sivanandan, Pulkit Mathur, 2019

* <[Cross Domain Imitation Learning](https://arxiv.org/abs/1910.00105)> by Kun Ho Kim, Yihong Gu, Jiaming Song, Shengjia Zhao, Stefano Ermon, 2019.

* [POMDP] <[Learning Belief Representations for Imitation Learning in POMDPs](https://arxiv.org/abs/1906.09510)> by Tanmay Gangwani, Joel Lehman, Qiang Liu, Jian Peng, 2019.

* <[Adversarial Imitation Learning from Incomplete Demonstrations](https://arXiv.org/abs/1905.12310)> by Mingfei Sun and Xiaojuan Ma, 2019.

* <[Self-Improving Generative Adversarial Reinforcement Learning
 ](https://dl.acm.org/citation.cfm?id=3331673)> by Yang Liu, Yifeng Zeng, Yingke Chen, Jing Tang, Yinghui Pan, 2019.

* [GMMIL] <[Imitation Learning via Kernel Mean Embedding](https://www-users.cs.umn.edu/~hspark/mmd.pdf)> by Kee-Eung Kim, Hyun Soo Park, AAAI 2018.
 
* <[Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning
](https://arxiv.org/abs/1809.02925)> by Ilya Kostrikov, Kumar Krishna Agrawal, Debidatta Dwibedi, Sergey Levine, Jonathan Tompson, 2018.

* [HRL] <[Directed-Info GAIL: Learning Hierarchical Policies from Unsegmented Demonstrations using Directed Information
](https://arxiv.org/abs/1810.01266)> by Ziyu Wang, Josh Merel, Scott Reed, Greg Wayne, Nando de Freitas, Nicolas Heess, 2018.

* <[Robust Imitation of Diverse Behaviors
](https://arxiv.org/abs/1707.02747)> by Arjun Sharma, Mohit Sharma, Nicholas Rhinehart, Kris M. Kitani, 2017.

* <[End-to-End Differentiable Adversarial Imitation Learning
 ](http://proceedings.mlr.press/v70/baram17a.html)> by Nir Baram, Oron Anschel, Itai Caspi, Shie Mannor, 2017.

* [InfoGAIL] <[InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations](http://papers.nips.cc/paper/6971-infogail-interpretable-imitation-learning-from-visual-demonstrations)> by Yunzhu Li, Jiaming Song and Stefano Ermon, 2017.
  * Propose **InfoGAIL**, learn interpretable (diverse) policies with mutual information technique (See InfoGAN).

* [GAIL] <[Generative Adversarial Imitation Learning](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning)> by Jonathan Ho and Stefano Ermon, 2016.
  * Propose **GAIL**, minimize the JS divergence of policy and expert policy with GAN's technique.
  
* [TRAIL]<[Task-Relevant Adversarial Imitation Learning](https://arxiv.org/abs/1910.01077)>, by Konrad Zolna, Scott Reed, Alexander Novikov, Sergio Gomez Colmenarej, David Budden, Serkan Cabi, Misha Denil, Nando de Freitas, Ziyu Wang, 2019.

* [AGAIL]<[Adversarial Imitation Learning from Incomplete Demonstrations](https://arxiv.org/abs/1905.12310)>, M. Sun et al., 2019
  
## Support-Estimation

Recently, there is a paper designs a new idea for imitation learning, which learns a fixed reward signal which obviates the need for dynamic update of reward functions.

* <[Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation](https://arxiv.org/pdf/1905.06750)> by Ruohan Wang, Carlo Ciliberto, Pierluigi Amadori, Yiannis Demirisn, 2019.
  * Propose **RED**, use RND for IL.

## Multi-Agent

* <[Coordinated multi-agent imitation learning](https://dl.acm.org/citation.cfm?id=3305587)> by Dylan Hadfield-Menell, Stuart J. Russell, Pieter Abbeel and Anca Dragan, NIPS 2016.

## MA-Inverse-RL

* <[Cooperative Inverse Reinforcement Learning](http://papers.nips.cc/paper/6420-cooperative-inverse-reinforcement-learning)> by 	Hoang M. Le, Yisong Yue, Peter Carr, Patrick Lucey,	ICML 2017.

* <[Multi-Agent Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1907.13220)> by Lantao Yu, Jiaming Song, Stefano Ermon. ICML 2019.

* <[Comparison of Multi-agent and Single-agent Inverse Learning on a Simulated Soccer Example](https://arxiv.org/pdf/1403.6822.pdf)> by Lin X, Beling P A, Cogill R. arXiv 2014.

* <[Multi-agent inverse reinforcement learning for zero-sum games](https://arxiv.org/pdf/1403.6508.pdf)> by Lin X, Beling P A, Cogill R. arXiv, 2014.

* <[Multi-robot inverse reinforcement learning under occlusion with interactions](http://aamas2014.lip6.fr/proceedings/aamas/p173.pdf)> by Bogert K, Doshi P. AAMAS 2014.

* <[Multi-agent inverse reinforcement learning](http://homes.soic.indiana.edu/natarasr/Papers/mairl.pdf)> by Natarajan S, Kunapuli G, Judah K, et al. ICMLA 2010.

## MA-GAIL

* <[Multi-Agent Generative Adversarial Imitation Learning](https://papers.nips.cc/paper/7975-multi-agent-generative-adversarial-imitation-learning)> by Jiaming Song, Hongyu Ren, Dorsa Sadigh, Stefano Ermon. NeurIPS 2018.

## Applications

* <[Better-than-Demonstrator Imitation Learning via Automatically-Ranked Demonstrations](https://arxiv.org/pdf/1907.03976.pdf)>, by Daniel S. Brown, Wonjoon Goo, Scott Niekum, CoRL 2019.

* <[Multi-Task Hierarchical Imitation Learning for Home Automation](http://ronberenstein.com/papers/CASE19_Multi-Task%20Hierarchical%20Imitation%20Learning%20for%20Home%20Automation%20%20.pdf)>, by Fox Roy, Berenstein Ron, Stoica Ion and Goldberg Ken, 2019.

* <[Imitation Learning for Human Pose Prediction](https://arxiv.org/pdf/1909.03449.pdf)>, by Borui Wang, Ehsan Adeli, Hsu-kuang Chiu, De-An Huang and Juan Carlos Niebles, 2019.

* <[Making Efficient Use of Demonstrations to Solve Hard Exploration Problems](https://arxiv.org/abs/1909.11821)>, by Tom Le Paine, Caglar Gulcehre, Bobak Shahriari, Misha Denil, Matt Hoffman, Hubert Soyer, Richard Tanburn, Steven Kapturowski, Neil Rabinowitz, Duncan Williams, Gabriel Barth-Maron, Ziyu Wang, Nando de Freitas and Worlds Team, 2019.

* <[Imitation Learning from Video by Leveraging Proprioception](https://arxiv.org/pdf/1905.09335.pdf)>, by Faraz Torabi, Garrett Warnell and Peter Stone, IJCAI 2019.

* <[End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)>, by Felipe Codevilla, Matthias Müller, Antonio López, Vladlen Koltun, Alexey Dosovitskiy, ICRA 2018.

* <[End-to-End Learning Driver Policy using Moments Deep Neural Network](https://ieeexplore.ieee.org/abstract/document/8664869)>, Qian Deheng, Ren Dongchun, Meng Yingying, Zhu Yanliang, Ding Shuguang, Fu Sheng, Wang Zhichao and Xia Huaxia, ROBIO 2018.

* <[R2P2: A ReparameteRized Pushforward Policy for Diverse, Precise Generative Path Forecasting](https://link.springer.com/chapter/10.1007/978-3-030-01261-8_47)>, by Qian Deheng, Ren Dongchun, Meng Yingying, Zhu Yanliang, Ding Shuguang, Fu Sheng, Wang Zhichao and Xia Huaxia, ECCV 2018. [[blog]](http://www.cs.cmu.edu/~nrhineha/R2P2.html) 

  * <[Learning Montezuma’s Revenge from a Single Demonstration](https://arxiv.org/pdf/1812.03381.pdf)>, by Tim Salimans and Richard Chen, 2018.
  
  * <[ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/pdf/1812.03079.pdf)>, by Mayank Bansal, Alex Krizhevsky and Abhijit Ogale, 2018.
  
  * <[Video Imitation GAN: Learning control policies by imitating raw videos using generative adversarial reward estimation](https://arxiv.org/pdf/1810.01108.pdf)>, by Subhajit Chaudhury, Daiki Kimura, Asim Munawar and Ryuki Tachibana, 2018.

  * <[Query-Efficient Imitation Learning for End-to-End Autonomous Driving](https://arxiv.org/abs/1605.06450)>, by Jiakai Zhang and Kyunghyun Cho, 2016.

