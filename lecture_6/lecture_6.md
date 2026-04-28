# Lecture 6: Q-Learning

## Recap: Some Useful Objects

![6_1](./6_1.png)

## Recap: Methods

![6_2](./6_2.png)

## Recap: Off-Policy Policy Evaluation

![6_3](./6_3.png)

## The plan for today

**Value-based RL methods**

1. Q-learning method

   a. Policy iteration

   b. Bellman optimality equation

   c. How to collect data for Q-learning methods

2. Q-learning in practice

   d. Target networks

   e. Double DQN

   f. N-step returns

**Key learning goals:**

- How Q-functions relate to policies

- How to do RL without learning an explicit policy

- How to stabilize Q-learning in practice

## A thought exercise

Say you have some policy and for your policy you have a pretty good estimate of
the Q-function.

For some policy $\pi$, say you have an accurate estimate $\hat{Q}^\pi(s, a)$ for
all $s, a$.

**Recall: Definition of Q-values**

$$ Q^\pi(s_t,a_t) = \sum_{t'=t}^{T}{\mathbb{E}_\pi\left[r(s_{t'},a_{t'})|s_t,a_t\right]} $$

"total reward we get if we take $a_t$ in $s_t$... and then follow the policy
$\pi$"

And an object that we're going to be thining about a lot today is if you have
some estimate of the Q-function for, in this case a policy, then, one thing you
could do is, instead of following soome policy that you've already learned, you
could potentially formulate a policy by choosing the action that maximizes your
estimate. So instead you could pick your action as your highest p-value.

$$ \max_a\hat{Q}^\pi(s, a)  $$

To continue on this, we'll define a policy that does exactly that.

Define a new policy

$$
{\pi'}_{\text{new}}(a_t|s_t) =
\begin{cases}
1 & \text{if } a_t = \arg\max_a\hat{Q}_{\phi}^{\pi}(s_t, a) \\
0 & \text{otherwise}
\end{cases}
$$

And that policy has a probability of $1$ for taking the action that maximizes
the p-value, and has a probability of $0$ otherwise. So this is a deterministic
policy that always follows what the p-value is telling you is the best.

We'll look at this in the context of an example. Say that we have some 2D
navigation problems. We'll be starting somewhere in this blue box, and the goal
is to reach to the star.

![6_4](./6_4.png)

And let's say we can move in 2D. The reward is one at the tar, and 0 elsewhere,
it's very simple.

And let's say our current policy always goes to the right.

![6_5](./6_5.png)

As one additional point of detail, if you're going to move up, it takes one
timestep to move about this distance and it takes about two time steps to mvoe
this distance.

![6_6](./6_6.png)

So my question for you is if you define this new policy, you have this current
policy that always goes right. You have an accurate estimate of the Q-values for
that policy. If you define this new policy, is that better than the current
policy? Is it worse than the current policy? Is it the same as the current
policy? Is it the optimal policy? And why or why not?

![6_7](./6_7.png)

So let's establish what is happening, consider this representation on the board:

![6_8](./6_8.png)

The old policy is going right, and so that is not optimal because if you're
starting here, then you're not going to hit the star. If you think about what
$Q^\pi$ is for the old policy, kind of everything in this region right here:

![6_9](./6_9.png)

Is going to have a $Q^\pi(s, \rightarrow) = 1$ for all of these states and
actions, where $\rightarrow$ indicates the action $a$ is "moving right". So, for
example, if the action moves upwards, it might actually miss the star if it
follows the policy "move right" after taking that action.

And then if you think of what $Q^\pi$ starts to look like right here (one cell
below), the $Q^\pi(s, \rightarrow) = 0$ because we won't arrive at the star.
Now, what is the $Q$-value if from this position, we take the action of going
up? It is $Q^\pi(s, \uparrow) = 1$. This is because if we go up by one cell,
then follow the policy "go right", we will arrive at the star.

Lastly, if we go further down one row further down, if we have
$Q^\pi(s, \uparrow)$, we will get $Q^\pi(s, \uparrow) = 0$, because if go up and
then follow the policy, we will not arrive at the star.

So what this is going to do is it's going to be going right on the middle row
(directly to the star), and for the row directly below it, it will be going up,
because it's going to be choosing the action that has the maximum value of
$Q^\pi$. Then for the row below that one, the $Q-values$ are $0$ because all of
the $Q$-values for the next row, the following of the policy actions (go right)
are also $0$ (you can't reach the star by going up one and then following the
policy of going right.).

![6_10](./6_10.png)

So, let's now ask the question: Is this the optimal policy?

So, the new policy is better than the old one (go up one and then go right
instead of just go right), but it's not the optimal one (it doesn't get the star
in all/most cases). So, if we iterate it a couple times though, it will (through
iteration, it will learn the basic topography of this matrix and adjust the
policy based off it's position in the matrix).

So the takeaway here is that this new policy will always be at least as good as
the old policy, assuming that $Q^\pi$ is accurate:

$$
{\pi'}_{\text{new}}(a_t|s_t) =
\begin{cases}
1 & \text{if } a_t = \arg\max_a\hat{Q}_{\phi}^{\pi}(s_t, a) \\
0 & \text{otherwise}
\end{cases}
$$

**Can we omit poilcy gradient completely?**

$Q^\pi(s_t, a_t)$: expected reward from taking $a_t$ and subsequently following
$\pi$

This is where we can start to think about: can we actually start omitting
policies?

$\arg \max_{a_t}Q^\pi(s_t, a_t)$: best action from $s_t$, if we then follow
$\pi$ afterwards.

So, $Q$ is the expected reward when we do this $\arg max$, we'll be taking the
best action from state $s$ if you then follow the policy $\pi$ afterwards.

And this is going to be at _least_ as good as any other action that the policy
would have taken, $a_t \sim \pi(a_t, s_t)$, _regardless_ of what the policy,
$\pi(a_t,s_t)$, is.

You can see that because it's taking the action that has the maximum expressive
reward.

This essentially indicates that we should forget policies, let's just do this!
This is done rather than actually explicitly learning a NN for the policy.

$$
{\pi'}_{\text{new}}(a_t|s_t) =
\begin{cases}
1 & \text{if } a_t = \arg\max_a\hat{Q}_{\phi}^{\pi}(s_t, a) \\
0 & \text{otherwise}
\end{cases}
$$

And so what this looks like as an iterative algorithm is we would run our policy
to collect data, then we would fit a model to estimate $Q^\pi$ according to that
latest batch of data. You could also do this in an off-policy way, like we
talked about in the last lecture. And then to improve your policy, you can just
define the policy to be the policy that maximizes your $Q$-value. And then, once
you define that new policy, you can collect some new data and then estimate
again $Q^\pi$, but in this case now $Q^\pi$ is your new policy and not your old
policy, and so on.

1. Run policy to collect batch data

2. Fit model to estimate expted return (Estimate $Q^\pi$)

3. Improve policy ($\pi \leftarrow \pi'$)

4. Repeat

Note again that the new policy is equal to the piecewise expression from
earlier. So we're not learning a NN for the policy $\pi$. We're not explicitly
representing it as a model, but it's helpful to have some notation to refer to
the system. The only model is the $Q$-function.

$$
{\pi'}_{\text{new}}(a_t|s_t) =
\begin{cases}
1 & \text{if } a_t = \arg\max_a\hat{Q}_{\phi}^{\pi}(s_t, a) \\
0 & \text{otherwise}
\end{cases}
$$

---

Q&A: 1

This is going to be easiest when you have a discrete set of actions that you
might take, like going up, going to the right, etc. If you have continuous
actions you actually can still use this algorithm. Although you need to figure
out how to perform that max operation. There's a couple ways you could
conceivably run the $\arg \max$ operation. In principle, you could use something
like gradient descent, although that turns out not to work very well. There's a
variety of reasons why that won't work very well. If your actions are not very
multidimensional, it might be better to simply try to sample a bunch of
different actions and pick the one that has the highest $Q$-value. There's other
sampling based optimization algorithms where you can iteratively sample and
refine your estimates from there.

If you have a discrete action space that's like text for example you could, in
principle, do some sort of discrete search over your things where the $Q$-value
is what your trying to maximize.

---

Q&A: 2

How does this work in cases where the rewards are very sparse?

We saw one example where the rewards are sparse, and it can still work in this
scenario. The thing that you need to be aware of when the rewards are sparse is
that you may need to do this loop of resetting the policy potentially a large
number of times because each time you do this. Like, the first time you do this,
you're actually only getting a good policy here (from the starting point of the
matrix), and if the matrix was much bigger, that would mean you need to keep on
doing it as you go down, and that is often referred to as a backup. You're sort
of backing up the reward from here (at the star) to the other state. And you'll
need to do those backups for as long as the horizon of the problem is. You need
to make sure you're doing many iterations.

**From actor-critic to critic only**

Off-policy actor critic with replay buffer

1. take action $a \sim \pi(a|s)$, get $s, a, s', r$, store in $\mathscr{R}$

2. sample a batch $\{s_i, a_i, r_i, s_{i}'\}$ from buffer $\mathscr{R}$

3. update $\hat{Q}_{\phi}^{\pi}$ using targets
   $y_i = r_i + \gamma\hat{Q}_{\phi}^{\pi}(s_{i}', a_{i}')$ where
   $a_{i}' \sim \pi(\cdot|s_{i}')$

4. $\nabla_\theta J(\theta) \approx \dfrac{1}{N}\sum_{i}{\nabla_\theta\log\pi_\theta(a_{i}^{\pi}, s_i)\hat{Q}^\pi(s_i, a_{i}^\pi)}$
   where $a_{i}^\pi \sim \pi_\theta(a|s_i)$

5. $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$

6. Repeat

So from this off-policy actor critic with replay buffer, we actually can remove
a few steps and add just one step to get closer to our algorithm for Q-learning.

Q-learning

1. take action $a \sim \pi(a|s)$, get $s, a, s', r$, store in $\mathscr{R}$

2. sample a batch $\{s_i, a_i, r_i, s_{i}'\}$ from buffer $\mathscr{R}$

3. update $\hat{Q}_{\phi}^{\pi}$ using targets
   $y_i = r_i + \gamma\hat{Q}_{\phi}^{\pi}(s_{i}', a_{i}')$ where
   $a_{i}' \sim \pi(\cdot|s_{i}')$ (policy evaluation) (Note: can do multiple
   gradient steps here)

4. define new Policy (policy improvement)

$$
\pi(a_t|s_t) =
\begin{cases}
1 & \text{if } a_t = \arg\max_a\hat{Q}_{\phi}^{\pi}(s_t, a) \\
0 & \text{otherwise}
\end{cases}
$$

5. Repeat

This is often called "Policy iteration".

![6_11](./6_11.png)

To write this out on the board we have:

0. Collect data from $\pi$

1. Fit $\hat{Q}^\pi(s, a)$ using the target from the last section
   $y_i = r(s,a) + \gamma Q(s', a')$ Where $r(s,a)$ is sampled from the buffer,
   and $a'$ is sampled form the policy $a' \sim \pi(\cdot|s')$

2. Improve our policy:

$$
\pi(a_t|s_t) =
\begin{cases}
1 & \text{if } a_t = \arg\max_a\hat{Q}_{\phi}^{\pi}(s_t, a) \\
0 & \text{otherwise}
\end{cases}
$$

---

Q/A:

Is this policy deterministic?

Yes, this policy is deterministic. This is actually one reason why you might not
want to collect data from this policy because you're not going to have many data
states in your actions. And we'll talk in a couple slides about what policy you
might want to collect data from instead.

---

Why are we sampling from $a'$ and not an average over actions?

There's a couple things here. First, you can do an average and sample multiple
times and that actually will give you a better estimate of the target value. The
second thing is instead of sampling this from $\pi$,
$a_{i}' \sim \pi(\cdot|s_{i}')$, you can actually do this improvement step in
the $Q$-function target itself. You can actually define it this way:

$$ \underbrace{y_i = r_i + \gamma\max_{a'}\hat{Q}_{\phi}(s_{i}',a')}_{\text{Q-values for your new policy!}} $$

![6_12](./6_12.png)

You can think of this as already putting the improvement step in your
$Q$-function update, and you're actually learning the $Q$-values associated with
the new policy that you're going to be defining.

---

Q-learning

1. take action $a \sim \pi(a|s)$, get $s, a, s', r$, store in $\mathscr{R}$

2. sample a batch $\{s_i, a_i, r_i, s_{i}'\}$ from buffer $\mathscr{R}$

3. update $\hat{Q}_{\phi}^{\pi}$ using targets
   $y_i = r_i + \gamma\max_{a'}\hat{Q}_{\phi}(s_{i}',a')$

4. define new Policy (policy improvement)

$$
\pi(a_t|s_t) =
\begin{cases}
1 & \text{if } a_t = \arg\max_a\hat{Q}_{\phi}^{\pi}(s_t, a) \\
0 & \text{otherwise}
\end{cases}
$$

5. Repeat

**Why does this make sense?**

Recall:

$$ Q^\pi(s,a) = r(s,a) + \gamma\mathbb{E}_{s' \sim p(\cdot|s,a),\overline{a}' \sim \pi(\cdot|s')}\left[Q^\pi(s',\overline{a})\right] \forall (s,a) $$

This equation is true for all states and actions. So we defined that the
$Q$-function is equal to the reward plus the function at the next state. This is
true for all states and actions. It's actually also true for any policy.

This holds for any policy $\pi$

Including the optimal policy. You can actually write down another equation that
is specifically for the optimal policy. In particular, we know that the optimal
policy is always going to be taking the actions that maximize our future
rewards.

The optimal plicy is always going to be the action that maximizes the $Q$-value.
And so we can write down an equation that holds just for the optimal policy,
which is the same equation as the one there, except instead of an expectation
over the next action, it's a maximization over the next action. So what that
looks like is:

$$ Q^{\pi^*}(s,a) = r(s,a) + \gamma\mathbb{E}_{s' \sim p(\cdot|s,a)}\left[\max_{a'}Q(s',a')\right] $$

We're defining the $Q$-function as the reward at the current time step plus the
sum of the future rewards. And for that sum of future rewards we have some
expectation over the next state that will happen. And because we know that the
optimal policy, or the best policy, is going to be taking the action that is
maximizing the $Q$-value, then we can write this as the max of $Q$ at the next
state.

---

Q/A?

Why are we talking all about $Q$ and why not $V$?

So you actually can write down all of these equations for $V$ in a very similar
way. The thing that's very nice about $Q$ is that we can actually get a policy
out of $Q$. So if we know what $Q$ is, then we can do this $\arg \max_{a}$ over
actions to get a policy that's better than our previous policy. Whereas with the
value function, if we only know $V(s)$, then we don't know which action will
lead to higher rewards. In principle, with the value function, you could try to
do some sort of look ahead. If you have a sense of what the dynamics are, you
could then try to predict what $s'$ is, and then try to pick actions that are
maximizing that. What that would look like is:

$$ \max_{a}\mathbb{E}_{s'\sim p(\cdot|s,a)}V(s') $$

You could, in principle, try to do something like this, but we don't know what
these dynamics, $s'\sim p(\cdot|s,a)$, are. And we can't necessary predict what
the next state will be as a consequence of our actions. So in general it's a lot
harder to get a policy out of a value function, and to improve our policy
ultimately, with a value function than it is with a $Q$-function.

---

So now going back to this equation, this is true for the optimal policy and this
is for $Q^{\pi^*}$, where I'm writing $\pi^*$ as the optimal policy.

$$ Q^{\pi^*}(s,a) = r(s,a) + \gamma\mathbb{E}_{s' \sim p(\cdot|s,a)}\left[\max_{\overline{a}'}Q^{\pi^*}(s', \overline{a}')\right] $$

And if we look at our update step equation here:

$$ y_i = r_i + \gamma\max_{a'}\hat{Q}_{\phi}(s_{i}',a') $$

We're trying to fit a $Q$-function that is predicting this. Essentially with the
optimal $Q^{\pi^*}$ equation, you can think about it as this update, as we are
trying to make this equation true.

$$ Q^{\pi^*}(s,a) = r(s,a) + \gamma\mathbb{E}_{s' \sim p(\cdot|s,a)}\left[\max_{\overline{a}'}Q^{\pi^*}(s', \overline{a}')\right] \forall (s,a) $$

So we're trying to make the left side of the equation tot he right side of the
equation. And when this equation is true, it means that our policy is the
optimal one.

So essentially one way to think about $Q$-learning is that when we're optimizing
step 3, we're trying to find a $Q$-function such that it satisfies this
equation, which is only true for the optimal policy.

---

Q/A:

Is this an "if and only if" condition?

This is an "if and only if" condition. This only holds true for the optimal
policy and if you have the optimal policy, then this must be true.

---

If we have really good estimates of $Q$, will this converge? We'll talk about
this in one minute, bu t first let's talk about some terminology real quick.

RL has a ton of terminology, it's useful to know some of it so that if someone's
talking to you about RL or you're reading a paper, you know what they're talking
about. This first equation is often referred to as the
[Bellman Equation](https://en.wikipedia.org/wiki/Bellman_equation). It's a very
useful object. It is what we were using to try to do policy evaluations for a
policy:

$$ Q^\pi(s,a) = r(s,a) + \gamma\mathbb{E}_{s' \sim p(\cdot|s,a),\overline{a}' \sim \pi(\cdot|s')}\left[Q^\pi(s',\overline{a})\right] \forall (s,a) $$

And the second equation, it's called the
[Bellman Optimality Equation](https://www.sciencedirect.com/topics/engineering/bellman-optimality-equation),
because it's specifically for optimal policy:

$$ Q^{\pi^*}(s,a) = r(s,a) + \gamma\mathbb{E}_{s' \sim p(\cdot|s,a)}\left[\max_{\overline{a}'}Q^{\pi^*}(s', \overline{a}')\right] \forall (s,a) $$

Sometimes, people might informally call the second one a Bellman equation as
well. But, yeah, officially, the first one is the bellman equation, and the
second one is Belman Optimality. You can derive ones that are equations that are
also for the value function $V$ instead of the $Q$ function, and people often
also refer to those as Bellman equations and Bellman Optimality equations.

![6_13](./6_13.png)

---

Alright, so getting back to the question: If we do this iteratively, will we
converge to the optimal policy?

So, there's good news and bad news here.

The good news is that if you're in a very simple setting where you can actually
store all of your key values in a table for every single state and action. So
this would be for a very small discrete state and action space. And if you have
sufficient exploration, then this algorithm will converge. So that's the good
news.

The bad news is that basically in any other scenario, it is not guaranteed to
converge. You can construct scenarios when it diverges even with just linear
functions, which is kind of sad. The flip side of that is that even though it
can diverge in scenarios, it can still be made to work very well, and we'll see
some examples of it working very well at the end of the lecture.

**Will this algorithm converge to optimal $Q^{\pi^*}$?**

Yes, if you maintain a table of $Q$-values for every state and action. More
generally, no.

Can construct scenarios where it diverges, evne with linear $Q$. _But_, it can
be made to work well.

**Note:** Q-learning is **off-policy**

In particular this optimization equation holds for all state and actions, and
that means that we don't have to have the actions coming from our current
policy. In particular, it might make a lot of sense for it to be broader than
our current policy. Because we're going to be doing this maximization over our
$Q$-values. So when we do this maximization, we're going to be considering a lot
of different possible actions. If we have a discrete action space, we might
actually just consider all of the possible actions, and when we do that, we
would want all of our $Q$-values to be accurate when we're considering all of
them. And if there's one action that we haven't collected data for that we're
going to be considering in this optimizaation, it might be erroneously high
because we didn't collect any data for it, and then you'll get an inaccurate
target value when you are fitting.

So, instead of taking collecting data from our current policy, it'd be useful to
collect data from some exploration policy that's specifically a policy that's
collecting a broader set of actions than our deterministic policy.

![image 6_14](./6_14.png)

So we want covered for a lot of different actions specifically for this term at
step 3, $a$. And there's a couple of choices here. One choice is to mostly
follow our policy most of the time, but with some small probability take a
completely random action. And so, what that would look like is:

$$
\pi_{\text{exploration}} =
\begin{cases}
\text{random action} & \text{w/ prob } \epsilon \\
\pi(\cdot|s)& \text{w/ prob } 1 - \epsilon
\end{cases}
$$

A policy that with some probability, and say take a random action with
probability $\epsilon$, and follow, maybe we call this
$\pi_{\text{exploration}}$, and follow your current policy with probabiliyt
$1 - \epsilon$.

And you can write this out slightly more formally as:

$$
\pi(a_t|s_t) =
\begin{cases}
1 - \epsilon & \text{if } a_t = \arg \max_{a_t}Q_{\phi}(s_t,a_t) \\
\epsilon/ \left(|\mathscr{A}| - 1\right) & \text{otherwise}
\end{cases}
$$

This can be a good choice, because it means that when you take a action
uniformly at random, it means that you'll get very good coverage of the actions
that you consider. And oftentimes, as you proceed through training, you might
start with a larger $\epsilon$, so that you have more exploration at the
beginning of training, and then as your policy is getting better and as you're
getting a better estimate of your $Q$-function, you can then make this smaller
and explore less as you go through things.

This is called "epsilon-greedy" because you are being greedy some of the time
and epsilon probability of the time, you're not being greedy and just taking a
risk.

Okay, and then another choice that is also somewhat reasonable, although a
little bit more complicated is to take actions with probability that is
proportional to the $Q$-values. And so if you have actions that have a higher
$Q$-value, then take those actions more frequently, and if you have actions with
lower probability, then take those actions less frequently.

So one way to write this is:

$$ \pi(a_t|s_t) \propto \exp(Q_{\phi}(s_t, a_t)) $$

So you could exponentiate your $Q$ values so that they're all positive, and then
normalize in orer to get a probability distribution over actions.

This is known as
[Boltzmann exploration](https://en.wikipedia.org/wiki/Boltzmann_machine).

![image 6_15](./6_15.png)

## Putting it together

full Q-learning with replay buffer:

1. collect data $\{(s_i, a_i, s_{i}', r_i)\}$ using some policy, add it to
   $\mathscr{R}$

   a. sample a batch $(s_i, a_i, s_{i}', r_i)$ from $\mathscr{R}$

   b.
   $\phi \leftarrow \phi - \alpha\sum_{i}{\dfrac{dQ_\phi}{d\phi}(s_i, a_i)\left(Q_\phi(s_i, a_i) - \left[r(s_i, a_i) + \gamma\max_{a'}Q_\phi(s_{i}', a')\right]\right)}$

   c. Repeat a and b multiple times, $K$ times.

2. After taking $K$ gradient steps, repeat 1.a.

![image 6_16](./6_16.png)

## How to make Q-learning stable?

1. collect data $\{(s_i, a_i, s_{i}', r_i)\}$ using some policy, add it to
   $\mathscr{R}$

   a. sample a batch $(s_i, a_i, s_{i}', r_i)$ from $\mathscr{R}$

   b.
   $\phi \leftarrow \phi - \alpha\sum_{i}{\dfrac{dQ_\phi}{d\phi}(s_i, a_i)\left(Q_\phi(s_i, a_i) - \left[r(s_i, a_i) + \gamma\max_{a'}Q_\phi(s_{i}', a')\right]\right)}$

   c. Repeat a and b multiple times, $K$ times.

2. After taking $K$ gradient steps, repeat 1.a.

So now let's see how Q-learning looks in practice.

We've talked about how Q-learning can diverge, and in general when you run
Q-learning with NN, it can be unstable, and one of the reasons for that is you
have this moving target in step 3:

$$ \underbrace{\left[r(s_i, a_i) + \gamma\max_{a'}Q_\phi(s_{i}',a_{i}')\right]}_{\text{this is a moving target}} $$

And here it seems that our optimization process is very non-stationary,
especially if $K$ is $1$. One trick is to try to slow down how often you change
the target. So, instead of having these targets change every single time you
change the parameters, one idea is that you could freeze the parameters of your
target network, and only update them periodically, only update them every 100
gradient steps, or something like that.

Can we change the target Q-values more slowly?

Simple idea: freeze parameters used for the target Q-values, update
periodically.

What this looks like is we can save a separate NN, another copy of our two
function parameters. We'll call this phi prime: $\phi'$. And, when we save this
out, when we, just for this target, we can use these new parameters, $\phi'$ in
our target.

These parameters will be frozen, and so it means that inside this inner loop,
the labels will not change. And, so this inner loop is essentially going to be
supervised learning with some data being added to the replay buffer. So this
should be a lot more stable. Supervised learning is a lot more stable generally.
Then, only periodically then update our target network parameters to be the same
as our latest function.

![image 6_17](./6_17.png)

Q-learning with replay buffer and target network:

0. save target network parameters: $\phi' \leftarrow \phi$

   $1$. collect data $\{(s_i, a_i, s_{i}', r_i)\}$ using some policy, add it to
   $\mathscr{R}$

   $2$. sample a batch $(s_i, a_i, s_{i}', r_i)$ from $\mathscr{R}$

   $3$.
   $\phi \leftarrow \phi - \alpha\sum_{i}{\dfrac{dQ_\phi}{d\phi}(s_i, a_i)\left(Q_\phi(s_i,a_i) - \underbrace{\left[r(s_i,a_i) + \gamma \max_{a'}\underbrace{Q_{\phi'}(s_{i}', a_{i}')}_{\text{frozen target network}}\right]}_{\text{labels don't change in inner loop.}}\right)}$

So as you can see, we have an oute r loop in which we loop through steps 0-3. We
have a First inner loop going from 1-3, which we repeat $N$ times, and we have
an inner loop going from 2-3, which we repeat $K$ times.

During the first inner loop going from 1-3, which we repeat $N$ times, we are in
an **inner loop, which is doing some supervised learning.**

During the second inner loop from 2-3, which we repeat $K$ times, we freeze our
target network $\gamma\max_{a'}Q_{\phi'}(s_{i}',a_{i}')$, and the **labels don't
change in the inner loop**,
$\left[r(s_i,a_i) + \gamma \max_{a'}Q_{\phi'}(s_{i}',a_{i}')\right]$.

This corresponds to what is known as a
[Deep Q Network](https://www.geeksforgeeks.org/deep-learning/deep-q-learning/)
algorithm, or a DQN. algorithm. It was introduced in 2013, and basically it was
the first DRL (Deep Reinforcement Learning) method (in my opinion).

What they found is that you could actually train NNs to play a variety of
different Atari games. It was one of the first examples of using RL to train NN
to do a pretty complicated task. They were actually training directly from the
pixel observations from the game.

Prior to that, a lot of the RL examples were on very toy settings, very low
dimensional settings.

---

## Are the Q-values accurate?

So that's the first trick, use a target network that you only periodically
update. There's two more things that make Q-learning work a lot better.

The first one is actually looking at the accuracy of the Q-values.

If you plot the reward over training, the reward goes up:

![image 6_18](./6_18.png)

That's good, this first graph shows a reward on the game of breakout, and the
second graph on the game of seaquest.

And if you also plot the Q-values over the course of training, they're also
going up. And that is definitely what you'd expect because the Q-values is
representing your average reward. So that's good. One thing to note, as a kind
of practical tip, is when you're looking at the loss function on your Q-values
during training, when you train with, the loss values will actually often go up
through training. This is sometimes a little bit alarming because you're used to
loss values going down. But the reason for that is you're collecting new data,
and that new data has higher $Q$-values and so the scaling of your loss function
is increasing throughout training. And at the very beginning, if you're getting
oneyear of rewrd everywhere, it might be very easy to get a loss function. So
don't be alarmed if your loss function is going up. Sometimes that means that
the algorithm or the policy is starting to.

$$ y_i = r(s,a) + \gamma \max{a'}Q(s',a') $$

![image 6_19](./6_19.png)

So our rewards are looking pretty good. As predicted, Q increases. So does the
return. You can also look at the alue over training, and you see that right in
the firstcase, the values are fairly uniform, but once you're in a state where
you really have to go up, then the Q-value for the action of up is higher than
going down or a no up. Likewise, it is important to go up, and so the Q-value is
higher for going upt hen the actions of going or no up. On the right side, it
doesn't matter, because the ball on the other side on the opponent's side, and
so the Q-value is more uniform. So when you visualize the Q-values, they look
pretty reasonable, which is good.

But then if you actually plot what the true returns are:

![image 6_20](./6_20.png)

The true returns are shown in the red here, on this red horizontal line. And the
estimate of the values according to the two networks are shown in this upper
line. And you can actually see this very huge gap between the estimate, where
the estimate is always a lot higher than the true value. So it sems like we're
seeing a lot of what's called overestimation, where our Q-function is
overestimating how good our policy is. And then there's this algorithm that was
introduced called
[Double DQN](https://en.wikipedia.org/wiki/Q-learning#Double_Q-learning) that's
able to get a much more accurate Q-function, it's able to overestimate far far
less than DQ.

Let's now talk a little bit about Double DQN and overestimations.

## Overestimation in Q-learning

$$ \text{target value } y_j = r_j + \gamma \max_{a_{j}'}Q_{\phi'}(s_{j}', a_{j}') $$

So our target value is where we're taking a max over an action, and when we take
this max, we're choosing an action that maximizes this $Q$ line:

$$ Q_{\phi'}(s_{j}', a_{j}') $$

Another way to write/think of this is, think of it as:

$$ Q_\phi(s,\arg \max_{a'}Q_\phi(s',a')) $$

And so we're both using an action that has high Q-value, and evaluating the
Q-value of that action, using the same Q-function. This means that if there's
noise in our Q-function, then it's very easy to basically pick actions that are
using that noise to get a higher Q-value.

$$ \text{target value } y_j = r_j + \gamma \max_{a_{j}'}\underbrace{Q_{\phi'}(s_{j}', a_{j}')}_{\text{this last term is the problem}} $$

$Q_{\phi'}(s',a')$ is not perfect - it looks "noisy"

hence $\max_{a'}Q_{\phi}'(s', a')$ _overestimates_ the next value!

note that
$\max_{a'}Q_{\phi'}(s',a') = Q_{\phi'}(s', \arg \max_{a'}Q_{\phi'}(s', a'))$

$$ \max_{a'}Q_{\phi'}(s',a') = \underbrace{Q_{\phi'}}_{\text{value also comes from } Q_{\phi'}}(s', \underbrace{\arg \max_{a'}Q_{\phi'}(s', a')}_{\text{action selected according to } Q_{\phi'}}) $$

![image 6_21](./6_21.png)

## Double Q-learning

$$ \max_{a'}Q_{\phi'}(s',a') = \underbrace{Q_{\phi'}}_{\text{value also comes from } Q_{\phi'}}(s', \underbrace{\arg \max_{a'}Q_{\phi'}(s', a')}_{\text{action selected according to } Q_{\phi'}}) $$

Now, if you instead use different Q-function for taking an action and evaluating
it, then you can decorrelate the noise. If 6ou had two different Q-functions,
they would have naturally different different noise and different errors.

So if we have a way to use different Q-values, we can actually then eliminate
this problem of overestimation by essentially decorrelating the noise, by using
a different Q-function to estimate the key axis here as to evaluate the axis.

![image 6_22](./6_22.png)

And so the key idea behind Double Q-learning is to have us use two different
networks. One network, network A for picking actions, and network B for
evaluating.

$$ Q_{\phi A}(s,a) \leftarrow r + \gamma Q_{\phi B}(s', \arg \max_{a'}Q_{\phi A}(s',a')) $$

$$ Q_{\phi B}(s,a) \leftarrow r + \gamma Q_{\phi A}(s', \arg \max_{a'}Q_{\phi B}(s',a')) $$

And then if they're noisy in different ways, then we're not going to then be
exploiting noise in our Q-values. Then, as a result not overestimating our
Q-values. This method is not perfect, but it does help.

![image 6_23](./6_23.png)

## N-step returns?

Recall that:

Q-learning target:
$y_{j,t} = r_{j,t} + \gamma\max_{a_j,t+1}Q_{\phi'}(s_{j,t+1},a_{j,t+1})$

$$ \underbrace{(s_{j,t+1},a_{j,t+1})}_{\text{these values are important if } Q_{\phi'} \text{ is good}} $$

$$ \underbrace{r_{j,t}}_{\text{these are the only values that matter if } Q_{\phi'} \text{ is bad!}} $$

If you just have a single reward function here, then that might be a bit
troublesome, because a single reward function might not tell you that much about
how good your Q-value is.

So, if you remember when we talked about active critic methods, we talked bout
using $n$-step return. So we use this Monte Carlo, using the sum of rewards, if
we use bootstrapping, we just do $r + V$ at each time step:

![image 6_24](./6_24.png)

And, using bootstreeping, instead we talked about this thing that we could do in
the middle where we summed the rewards over the next end timestep:

$$ \sum_{t}^{t+n-1}{r + V_{t+n}} $$

And then use the value at the timestep after that. We talked about how this has
less variance and lower bias. And lower bias in the sense that if your targets
are bad, the reward function is going to be more accurate.

So can we do this for Q-values?

What this would look like for Q-values is, say that your target was:

$$ Q(s_t,a_t)  $$

Then yhou define this as:

$$ y_i =r_t + r_{t+1} + \dots + r_{t+n-1} + \gamma \max_{a_{t+n}}Q(s_{t+n},a_{t+n}) $$

So this is what $n$-steps could look like for Q-learning.

One issue with this is that if you are doing this off-policy, then all these
$r$'s will be coming from the replay buffer, rather than the optimal policy,
what you're trying to estimate.

So let's look at some pros/cons:

- +(far) less biased target values when Q-values are inaccurate

- +typically faster learning, especially early on

- -these are the rewards for policy that collected the data

- -only actually correct when learning on-policy (not an issue when N=1)

**Ways to fix?**

Most commonly: ignore the problem & still use $N > 1$

Can also:

- dynamically choose $N$ to only use data that follows current policy (if data
  mostly on-policy, action space is small)

- use importance sampling

## When to use one online RL algorithm vs. another?

**Chelsea's advice**

**PPO & variants**

- When you care about stability, ease-of-use

- When you don't care about data efficiency

**DQN & variants**

- When you have discrete actions or low-dimensional continuous actions

**SAC & variants**

When you care most about data efficiency

When you are okay with tuning hyperparameters, less stability
