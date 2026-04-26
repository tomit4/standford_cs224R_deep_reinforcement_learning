# Lecture 3: Policy Gradients

## Recap

state $s_t$ - the state of the "world" at time $t$

action $a_t$ - the decision taken at time $t$

trajectory $\mathscr{t}$ - sequence of states/observations and actions

$$ (s_1, a_1, s_2, a_1, \dots , s_T, a_T) $$

reward function $r(s, a)$ - how good is $s$, $a$?

policy $\pi (a|s)$ or $\pi (a|o)$ - behavior, usually what we are trying to
learn

**Goal**: learn policy $\pi_{\theta}$ that maximizes _expected_ sum of rewards:

$$ \max_{\pi} \mathbb{E}_{T\sim p\theta(t)}\left[\sum_{t}^{T}{r(\mathbf{s}_t, \mathbf{a}_t)}\right] $$

where we have a probability distribution:

$$ P(T) = P(\mathbf{s}_1) \prod_{t=1}^T \pi(\mathbf{a}_t \mid \mathbf{s}_t)P(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t) $$

We learned about imitation learning last week with demonstrations $\mathscr{D}$

$$ \min_\theta - \mathbb{E}_{(s, a) \sim \mathscr{D}}\left[\log \pi_{\theta}(a | s)\right] $$

This means that if we have expert demonstrations you can try to imitate the
expert by trying to maximize the likelihood of the actions that the demonstrator
did. So, this is tryiing to actually produce a policy that leads to actions that
the expert would have taken. This is a very simple and scalable approach, but it
cannot outperform the demonstrator, and doesn't allow improvement through trial
and error.

We also introduced a couple definitions:

**Definitions**:

_offline_: using only an existing dataset, no new data from learned policy

_online_: using new data from learned policy

Today we will learn ways through RL to outperform the demonstrator, to improve
from practice, and specifically we'll be starting to talk about _online_ RL
algorithms.

## The plan for today

**Policy gradients**: our first online RL algorithm

1. On-policy policy gradient

   a. Derivation and intuition of policy gradients

   b. Full algorithm

   c. How to make it better - causality and baselines

2. Off-policy policy gradients

   a. Importance sampling

   b. KL constraints

**Key learning goals**:

- Key intuition behind policy gradients

- How to implement, when to use policy gradients

## Online RL Outline

1. Initialize the policy (randomly, with imitation learning, with heuristics)

For example, when training a NN, we need to initialize its weights. We can
initialize the weights of our policy randomly. We could also initialize it with
imitation learning, or use heuristics. It depends on the problem setting.

    a. Run policy to collect batch of data

    b. Use that batch of data to improve the policy

    c. Repeat a and b

## Evaluating the RL objective

Okay, so we talked about this reinforcement learning objective.

$$ \theta^{*} = \arg \max_\theta \mathbb{E}_{\mathscr{T}\sim p_\theta(\mathscr{T})}\left[\sum_{t}{r(s_t, a_t)}\right]$$

Say you have some plicy and you want to understand how good that policy is. Todo
that we can't just look at the weights of the NN, we actually have to run the
policy to understand how good it is.

And to do that we must estimate this expectation, which is a function of the
policy $J(\theta)$:

$$ \theta^{*} = \arg \max_\theta \underbrace{\mathbb{E}_{\mathscr{T}\sim p_\theta(\mathscr{T})}\left[\sum_{t}{r(s_t, a_t)}\right]}_{J(\theta)}$$

And what this means is that we just run the policy multiple times, and observe
the resulting reward, and then compute the average over the observed reward, and
that's an estimate of how good our policy is:

$$ J(\theta) = \mathbb{E}_{\mathscr{T}\sim p_\theta(\mathscr{T})}\left[\sum_{t}{r(s_t, a_t)}\right] \approx \frac{1}{N}\sum_{i}{\sum_{t}{r(s_{i, t}, a_{i, t})}} $$

Where $\sum_{i}$ above means that we sum over the samples from our policy
$\pi_\theta$.

**Can we get the gradient of the RL objective?**

Let's start in terms of trajectories.

$$ \theta^{*} = \arg \max_\theta \underbrace{\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t}{r(s_t, a_t)}\right]}_{J(\theta)}$$

So we have this trial and error process, and we improve our policy. Collecting
the data is straightforward, we've talked about how to run a policy to get a
rollout in the environment, we can sample an action, observe the net state, then
run that through our policy again to get a new action and repeat that process.
But then how do we then improve our policy after we've collected that data?

So we have our objective. We have some policy $\pi_\theta$, and the goal is to
maximize the expected rewards $\mathbb{E}$ when we sample from that policy. We
can consider the trajectgories that are sampled from the policy as being drawn
from distribution, written as:

$$ \tau \sim p_\theta(\tau) $$

This is because the distrubtion over trajectories is a function of the policy
that you're using. And our goal is to maximize the sum of rewards.

$$ \sum_{t}{r(s_1, a_1)} $$

For this demonstraton, we will summarize this simply as:

$$ r(\tau) $$

And again, our goal is to maximize our rewards with respect to our policy
$\theta$. This leaves us with the following mathematical expression:

$$ \max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[r(\tau)\right] $$

Now, how do we want to optimize this? We want to use gradient descent, because
we typically train NN with gradient descent. We need to figure out the gradient
of this objective (the expected sum) with regards to $\theta$. We will refer to
this term as $J(\theta)$

$$ \max_\theta \underbrace{\mathbb{E}_{\tau \sim p_\theta(\tau)}\left[r(\tau)\right]}_{J(\theta)} $$

With $J(\theta)$ representing the obective we are trying to maximize.

It is a little difficult to get a gradient of our sampling directly, because the
relationship that this has on $\theta$ is through the trajectgories that we
sampled. What this means is we can't just differentiate through this sampling
process, and that sampling process also depends on the dynamics of the "world",
which we also don't know.

So if we're going to think about getting a gradient of $J(\theta)$:

$$ \nabla_{\theta} J(\theta) $$

It is helpful to write down this expectation as an integral:

$$ \nabla J(\theta) = \nabla_\theta \int{p\theta(\tau)r(\tau)\, d\tau} $$

Because an integral is just a summation, we can bring the gradient inside the
integral. In particular, what that looks like is:

$$ \nabla J(\theta) = \int{\nabla_\theta p_\theta(\tau)r(\tau)\, d\tau}$$

Now, this doesn't really help us, because we don't really want to have to take
this integral in order to take a gradient. Now, there is a convenient trick that
we can use to try to put this into a form that's more useful. In particular, the
trick is that if you think about, we can take the log of both sides of (one of)
our integrand(s):

$$ p_\theta(\tau)\nabla_\theta\log p_\theta(\tau) = p_\theta(\tau)\nabla_\theta \log p_\theta(\tau) $$

Recall that the derivative of $\log(x)$ is $\dfrac{1}{x}$, so this becomes equal
to:

$$ \quad = p_\theta(\tau)\frac{}{p_\theta(\tau)} $$

And then we need also need to apply the chain rule of the other variables
($\nabla_\theta$):

$$ \quad = p_\theta(\tau)\frac{\nabla p_\theta(\tau)}{p_\theta(\tau)} $$

And as you can see, the $p_\theta(\tau)$ cancels out, leaving us with:

$$ \quad = \nabla_\theta p_\theta(\tau) $$

And because of this we can replace our gradient in our original integral from
before:

$$ \nabla_\theta J(\theta) = \int{p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)r(\tau)\, d\tau} $$

And now we can now represent $p_\theta(\tau)$ as an expectation, the first term.

$$ \quad = \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\nabla_\theta \log p_\theta(\tau)r(\tau)\right] $$

What we have done is used this log trick to get $p_\theta(\tau)$ out of the
integral, and then get an expectation, and we start to get something closer to
something we can actually evaluate.

In particular, this expectation $\mathbb{E}_{\tau\sim p_\theta(\tau)}$ is
something that we can estimate with samples, and now $\theta$ is actually
appearing on the inside of the summation
$\left[\nabla_\theta\log p_{\theta}(\tau)r(\tau)\right]$. What this means is
that we are now closer to something we can more easily evaluate.

We still have this term, $\log p_\theta(\tau)$, which still doesn't look like
$\pi_\theta(a | s)$, so we still need to work on this term, but we're getting
closer to something we can evaluate.

To address this term, let's first remember that:

$$ p_\theta(\tau) = p(s_1)\prod_t\pi(a_t|s_t)p(s_{t+1}| s_t, a_t) $$

Then when we take the log of $p_\theta(\tau)$, as we did earlier, we get:

$$ \log p_\theta(\tau) = \log p(s_1) + \sum_{t}{\log\pi(a_t|s_t) + \log p(s_{t+1}| s_t, a_t)} $$

Notice that if we then take the gradient of the log, $\theta$ does not affect
the evaluation nor the dynamics in any way.

$$ \nabla\log p_\theta(\tau) = \nabla_\theta\sum_{t}{\log\pi_\theta(a_t|s_t)} $$

And this is quite useful, because we can plug this into our previous formula and
evaluate.

Let's plug this in and see:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t}{\nabla_\theta\log\pi(a_t|s_t)}\right]r(\tau) $$

We can also write this as:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\left(\sum_{t}{\nabla_\theta\log\pi(a_t|s_t)}\right)\left(\sum_{t}{r(s_t,a_t)}\right)\right]$$

This final mathematical equation is what is known as the "vanilla" policy
gradient, or simply The Policy Gradient. This is a lot nicer than our original
version, as we can sample trajectories from our policy, compute the reward, and
also compute the gradient of each of the log of $\pi$ for each of the actions,
and then sum those together, multiply, and we can then get a gradient that we
can apply to the weights of our NN, that is _actually_ leading to higher reward.

This allows us to approximate our policy:

$$ \nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\left(\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}|s_{i, t})}\right)\left(\sum_{t=1}^{T}{r(s_{i, t}, a_{i, t})}\right)} $$

Where $N$ represents the number of samples we take from our trajectory. Note
that the higher $N$ is, the better an estimate you're going to get of the
expectation of the integral.

Once we evaluate this gradient that we're estimating with samples, we'll then
take a gradient step on our policy parameters.

$$ \theta \leftarrow \theta + \alpha\nabla_\theta J(\theta) $$

So the full agorithm now looks like:

Full agorithm:

1. sample $\{\tau^i\}$ from $\pi_\theta(a_t|s_t)$

2. $\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\left(\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}|s_{i, t})}\right)\left(\sum_{t=1}^{T}{r(s_{i, t}, a_{i, t})}\right)}$:
   Evaluating an estimate of the gradient using those samples.

3. $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$: Applying that to
   the weights

4. Repeat this process.

**What does the gradient mean?**

So we've gone in depth on the mathematics of this, but what is the intuition?

Let's inspect our gradient evaluatioin:

$$ \nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\left(\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}|s_{i, t})}\right)\left(\sum_{t=1}^{T}{r(s_{i, t}, a_{i, t})}\right)} $$

Look at this first term here:

$$ \frac{1}{N}\sum_{i=1}^{N}{\left(\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}| s_{i, t})}\right)} $$

Recall from our last lecture on imitation learning that our objective is to
minimize the negative log probability:

$$ \min_\theta - \mathbb{E}_{(s, a) \sim \mathscr{D}}\left[\log\pi_\theta(a|s)\right] $$

And if we take the gradient of this objective, we get:

$$ \nabla_\theta J_{BC}(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\left(\sum_{t - 1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}| s_{i, t})}\right)} $$

Notice that this first term of our Policy Gradient objective is the same as the
gradient of the imitation learning objective.

So we can think of this first term as basically "I'm going to try to increase
the likelihood actions in the online data." But then the second term:

$$ \left(\sum_{t=1}^{T}{r(s_{i, t}, a_{i, t})}\right) $$

This is essentially saying that we are weighting all of that by the reward of
that trajectory. Like saying "how good was that trajectory?"

Thusly you can think of Policy Gradient as being an implementation of the
Imitation Gradient, but weighted by the rewards seen in the data.

**Intuition**:

- Increase the likelihood of actions you took in high reward trajectories.

- Decrease the likelihood of actions you took in negative reward trajectories.

_i.e._ do more of the good stuff, less of the bad stuff.

You can also think of this as a formalization of "trial-and-error"

**Improving the gradient**

Using causality

$$ \nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\left(\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}|s_{i, t})}\right)\left(\sum_{t=1}^{T}{r(s_{i, t}, a_{i, t})}\right)} $$

There actually is a failry simple way to improve our gradient.

Consider that if you are in the middle of a trajectory, and you take an action,
that action has no influence on previous states. And as our current vanilla
policy gradient algorithm is, the rewards in all states of the trajectory during
the gradient update, it still progresses through the iteration regardless if the
rewards from the trajectory were good or bad.

One way we can adjust this is instead of starting our rewards summatioin at
$t=1$, we can start it at the initial trajectory itself.

$$ \nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}|s_{i, t})}\left(\sum_{t'=t}^{T}{r(s_{i, t'}, a_{i, t'})}\right)} $$

Note that we also moved our second summation inside the first summation.

This essentially means that our algorithm is only looking at _future_ rewards,
not current ones. In other words, it means that current acts can only affect the
future, and not affect the past.

Policy behavior at time $t$ does not affect rewards at time $t' < t$

**Introducing baselines**

We can also improve our gradient by introducing a constant:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\nabla_\theta\log p_\theta(\tau)\left(r(\tau) - b\right)\right] $$

Where the constant $b$ represents a mean of our rewards, called a "baseline":

$$ b = \frac{1}{N}\sum_{i=1}^{N}{r(\tau)} $$

This means that rewards that are better than average will increase the
likelihood of repeating that behavior, and rewards that are less than average
will decrease the likelihood.

You might be wondering why are we allowed to do this?

We can see the effect that this has on the gradient. What we can show is that if
you subtract a constant, in expectation, this has zero effect on the gradient.

$$ \mathbb{E}\left[\left(\nabla\log p\right)b\right] $$

It is possible to show that this expectation is equal to $0$, and therefore we
can indeed at it to our gradient formula to improve it, as it has no effect on
our expecation.

$$ \mathbb{E}\left[\left(\nabla\log p\right)b\right] \int{p(\tau)\nabla\log p(\tau)b \, d\tau} $$

$$ \quad = \int{\nabla_\theta p_\theta(\tau)b\, d\tau} $$

We can move $\nabla_\theta$ and $b$ outside of the integral because these are
constants.

$$ \quad = \nabla_\theta b\int{p_\theta(\tau)\, d\tau} $$

This integral is equal to $1$, and the gradient of $1$ is $0$. so:

$$ \quad = (0)b(1) $$

$$ \quad = 0 $$

Basically, because adding this constant has no effect on the expectation, this
means we can safely subtract this constant, which again, is the mean of the
rewards, to the calculation of rewards and it will improve our gradient formula
as we find the policy for Policy Gradient.

You can think of this as a way to reduce **variance** of our expecatation, and
it's an **unbiased** way to do that.

Fully written out, this looks like:

$$ \nabla_\theta J(\theta) \approx \sum_{i}\left(\sum_{t}\nabla_\theta\log\pi_\theta(a_{t, i}|s_{t, i})\right)\left(\sum_tr(s_{i, t}, a_{t, i}) - b\right) $$

The drawback to this approach is that sometimes the mean of the behavior can
consolidate the variance of our rewards, losing nuance of certain behaviors that
we still want to encourage, but perhaps not as strongly as a more encouraged
behavior.

This causes a sort of "flattening" out of our gradient on certain behaviors
which still want to be encouraged more than others, but instead they are
consolidated into a neutral mean of behaviors.

Ultimately, policy gradient is still noisy/high-variance, and is best used with
dense rewards, and large batches.

**How to implement policy gradients?**

Our gradient currently, in its full form is:

$$ \nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_{i, t}|s_{i, t})\left(\left(\sum_{t'=t}^{T}{r(s_{i, t'}, a_{i, t'})}- b\right)\right)}} $$

Computing our gradients (the first major term prior to the rewards summation)
individually is inefficient, because this would indicate that we would end up
calculating $N \cdot T$ backwards passes. This leads us to ask:

Can we use automatic differentiation on the full objective?

We can, it is possible to:

Implement "surrogate objective" whose gradient is the same as $\nabla J$

It is called a surrogate because it's not the same as the very first objective
of maximizing the expected rewards. It looks like:

$$ \tilde{J}(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}{\sum_{t=1}^{T}{\log\pi_\theta(a_{i, t}|s_{i, t}) \left(\left(\sum_{t'=t}^{T}{r(s_{i, t'}, a_{i, t'})}- b\right)\right)}} $$

You don't have to completely understand what is happening here, just know that
this is essentially implementing a surrogate objective and then use back
propagation and auto diffing software to just do one backward pass.

This part in particular:

$$ \log\pi_\theta(a_{i, t}|s_{i, t}) $$

Creates a weighted maximum likelihood, and is a Cross-entropy loss for discrete
action policy, a squared error for Gaussian policy.

**Summary for far**

**Estimating gradient of RL objective**

- log gradient trick

- weigh policy likelihood by future rewards

- subtract baseline (e.g. average reward)

- even with tricks, gradient is _noisy_

**First reinforcement learning algorithm**

- collect batch of data, improve policy by applying gradient, repeat

- formalizes trial-and-error learning

**Key intuition**: do more high reward stuff, less low reward stuff

**What else is troublesome about policy gradients?**

Latest version of our gradient:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\left(\sum_{t=1}^{T}{\nabla_\theta\log\pi_\theta(a_t|s_t)}\right)\left(\left(\sum_{t'=t}^{T}{r(s_{t'}, a_{t'})}\right) - b\right)\right] $$

Full algorithm:

1. sample $\{\tau^i\}$ from $\pi_\theta(a_t|s_t)$

2. compute $\nabla_\theta J(\theta)$ using $\{\tau^i\}$

3. $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$

4. repeat

Note that we sample from $p_\theta(\tau)$

And that we change $\theta$ at 3 of our algorithm, at
$\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$

This means that every single time that we take a single gradient step, we need
to recollect data to estimate our gradient. And this is not really all that
good, because usually we do like thousands and thousands of gradient steps on
NNs. So this is a problem. This is when our:

Vanilla policy gradient is **on-policy**.

It means that the gradient that we are measuring requires data using our exact
policy.

**Definitions:**

_on-policy_: update uses only data from current policy

_off-policy_: update can reuse data from other, past policies

So, then, our question can becomes:

**Can we derive an Off-policy version of policy gradient?**

Importance sampling

$$ J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[r(\tau)] $$

What if we want to use samples from $\overline{p}(\tau)$ (_e.g._ previous
policy).

So what if we want to go off-policy? And by this we mean just a little
off-policy, in other words, we want to take more than one gradient step on a
single batch of data.

To do this, we can use imporant sampling. Important sampling is an important
technique even used outside of RL as well.

It's sort of kind of thinking about this scenario where you want to estimate a
quantity, an expectation, where you don't actually sample from that
distribution.

$$ \mathbb{E}_{x \sim p(x)}\left[f(x)\right] $$

So here we are trying to express an expectation for $f(x)$ given a sample of $x$
from $p(x)$, but let's say we only can sample from another function, say $q(x)$.
Here, $q(x)$ is what is known as a **proposal** sample, our proposed samples to
use.

In other words, we have samples of our "old" policy, but we want to now take
expectations of our new policy, after we've taken a gradient step.

If we write down this expectation as an integral:

$$ \mathbb{E}_{x \sim p(x)}\left[f(x)\right] $$

$$ \quad = \int{p(x)f(x)\, dx} $$

As long as $q(x)$ has reasonable support, we can add in this term that equals
$1$:

$$ \quad = \int{\frac{q(x)}{q(x)}p(x)f(x)\, dx} $$

And now we actually can write this as an expectation with respects to $q$:

$$ \quad =  \mathbb{E}_{q(x)}\left[\frac{p(x)}{q(x)}f(x)\right] $$

Essentially we are sampling from $q(x)$, then taking the probability of $f(x)
and weighting the quantity by $\dfrac{p(x)}{q(x)}$.

This means that we can use on principal samples from our old policy $q(x)$,
weigh them by quantity $f(x)$ times the probability of the sample under our new
policy $\dfrac{p(x)}{q(x)}$

$$ J(\theta) = \mathbb{E}_{\tau \sim \overline{p}(\tau)}\left[\frac{p_\theta(\tau)}{\overline{p}(\tau)}r(\tau)\right] $$

**Importance sampling**

Using proposal distribution $q$

$$ \mathbb{E}_{x\sim p(x)}[f(x)] = \int{p(x)f(x)\, dx} $$

$$ \quad = \int{\frac{q(x)}{q(x)}p(x)f(x)\, dx} $$

$$ \quad = \int{q(x)\frac{p(x)}{q(x)}f(x)\, dx} $$

$$ \quad = \mathbb{E}_{x\sim q(x)}\left[\frac{p(x)}{q(x)}f(x)\right]$$

**Note**: Important for $q$ to have non-zero support for high probability
$p(x)$.

we've shown this in terms of $p(\tau)$, but how do we show this in terms of
$\pi_\theta$?

$$ \frac{p_\theta(\tau)}{\overline{p}(\tau)} = \frac{p(s_1)\prod_{t=1}^{T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)}{p(s_1)\prod_{t=1}^{T}\overline{\pi}(a_t|s_t)p(s_{t+1}|s_t,a_t)} = \frac{\prod_{t=1}^{T}\pi_\theta(a_t, s_t)}{\prod_{t=1}^{T}\overline{\pi}(a_t|s_t)} $$

IF we say want to update our latest policy $\pi_{\theta'}$ but we want to use
samples from $\pi_\theta$

$$ \nabla_{\theta'}J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[\left(\sum_{t=1}^{T}{\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)}\right)\left(\left(\sum_{t'=t}^{T}{r(s_{t'},a_{t'})}\right) - b\right)\right] $$

$$ \nabla_{\theta'}J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}\left(\sum_{t=1}^{T}{\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)}\right)\left(\left(\sum_{t'=t}^{T}{r(s_{t'},a_{t'})}\right) - b\right)\right] $$

$$ \nabla_{\theta'}J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\prod_{t=1}^{T}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}\left(\sum_{t=1}^{T}{\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)}\right)\left(\left(\sum_{t'=t}^{T}{r(s_{t'},a_{t'})}\right) - b\right)\right] $$

And this product can become very small or very large, for larger length of
trajectory $T$.

Thusly in practice, we will consider expectations _over timesteps_ instead of
trajectories.

$$ \nabla_{\theta'}J(\theta') \approx \frac{1}{N}\sum_{i=1}^{N}{\sum_{t=1}^{5}{\frac{\pi_{\theta'}(s_{i, t}, a_{i, t})}{\pi_\theta(s_{i, t}, a_{i, t})}\nabla_{\theta'}\log\pi_{\theta'}(a_{i, t}| s_{i, t})}\left(\left(\sum_{t'=t}^{T}{r(s_{i, t'}, a_{i, t'})}\right) - b\right)} $$

The only hitch as this policy gradient iterates over state-action pairs, rather
than actions given states:

$$ \frac{\pi_{\theta'}(s_{i, t}, a_{i, t})}{\pi_\theta(s_{i, t}, a_{i, t})} $$

But we can often, in practice, approximate this as being equal to $1$.

The full algorithm looks the same, but now we can take **multiple gradient
steps** on the same batch.
