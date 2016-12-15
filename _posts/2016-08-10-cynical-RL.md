---
layout: post
title: Cynical Reinforcement Learning
---

Sometimes the world of reinforcement learning is so dark and cynical.
Gridworlds like the one I plottet below are utilized to experiment and explore different core characteristics of learning agents.
Unfortunately they aren't as popular as they have been as a research tool ([this paper](https://research.facebook.com/publications/mazebase-a-sandbox-for-learning-from-games/) shows that gridworlds still are very relevant) they are still omnipresent in teaching because they have so small discrete state spaces and actual humans can actually understand the state representations (mostly just xy coordinates on a grid).
The example below  has the following rules(taken from the excellent Sutton and Barto's amazing [book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html) on RL).

*  The agent starts in the top left corner.
*  Each step gets an negative reward of **-1**
*  If the agent steps on one of the magenta cliff tiles on the top, there is a negative reward of **-100** for falling off the cliff and the agent has to start again from the top left corner.
*  The goal is to reach the top right corner. When reaching the top right corner, everything starts all over again.


  ![A world of pain and suffering!!](/img/cliff_map.png)

Sometimes it strikes me how dark, twisted and cynical this is, even though it's just a toy example. Basically it states

* **-1** for every step you make means nothing less than **living is suffering**
*  Even if you reach your goal you don't get any reward, just you're suffering ends till it starts all over again. You can't escape.
* Jumping off the cliff won't help you. You just have to start over again. There's no escape.

There's also a much more positive life lesson one could learn from $$\epsilon$$-greedy exploration policies. I'll definitely have to write about it soon, to balance out this post.
