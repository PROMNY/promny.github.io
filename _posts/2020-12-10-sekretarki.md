---
layout: post
title:  "Is she/he the one? Monte-Carlo approach to the secretary problem."
---
<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
        
In this post, I will try to present my thoughts about the secretary problem.
I will show some simple Monte-Carlo simulations and results. 

# Secretary problem, marriage problem, optimal stopping theory.

The original problem was formulated under the following assumptions:
- There is a single position to fill.
- There is a known number of candidates \\(n\\).
- The candidates, if seen all together, can be ranked from best to worst.
- The candidates are being interviewed sequentially in random order.
- Immediately after an interview, the interviewed applicant is either accepted or rejected, and the decision is irrevocable.
- The decision to accept or reject an applicant can be based only on the relative ranks of the applicants interviewed so far.

More on the original problem: [Secretary problem](https://en.wikipedia.org/wiki/Secretary_problem)

So far the best solution to this problem was given by F. Thomas Bruss and is called \\(1/e \\) strategy. In this approach, we should reject the first \\( n/e \\) (e is the base of the natural logarithm) candidates and then select the first better candidate than the best in the first group.

However, in this approach success is defined as selecting the best candidate from the set.

# Let's test it numerically then.
Let \\( n \\) be a number of candidates and \\(k \in (0,1) \\) the percent of candidate in the first group.

We can calculate the probability of choosing the best candidate by using the following algorithm:   
- Generate a list of random numbers with length \\( n \\) and assign it as `candidates`.
- Divide the list into two sublists (`candidates_A, candidates_B`) according to the value of \\(k\\).
- Find the highest value in `candidates` and assign it to `best_in_set`.
- Find the highest value in `candidates_A` and assign it to `best_in_first_group`.
- Find the first higher value in the second sublist that is higher than the `best_in_first_group`.
- If this value is equal to `best_in_set` then we count it as a success.

We can calculate the probability of success by running this algorithm \\(m\\) times. 
Then probability is equal to \\[P=\frac{successes}{m}.\\]
We can easily calculate it in python. 
```python
def secretary_probability_of_success(n, k, m):
    """
    returns probability of choosing the best candidate for given:
    n - number of candidates
    m - number of random stets
    k - percent of candidates in test group
    """

    successes = 0
    k_index = ceil(n*k)
    for _ in range(m):
        candidates = np.random.rand(n)
        candidates_A = candidates[:k_index]
        candidates_B = candidates[k_index:]

        best_in_set = np.amax(candidates)
        best_in_first_group = np.amax(candidates_A)

        for i in candidates_B:
            if i > best_in_first_group:
                if i == best_in_set:
                    successes += 1
                break

    return float(successes / m)
```
For \\(n=100\\) and \\(m=10^6\\) we got a following result:

<div style="text-align:center">
<img src="/assets/sekretarki/1.png" alt="secretary" width="600"/>
</div>

As you can see that result is in agreement with the previous theory.

# Are we really looking for the best one?

Let's now assume that we are not looking for the best candidate. Instead, we may try to maximize the relative value of a chosen candidate.
We can modify the `secretary_probability_of_success` so that it averages the relative value of the chosen candidate.
There are two possibilities to count failures. In the first scenario `secretary_average_zero` we will assume that no candidate was selected so \\(0\\) will be added.
In the second scenario `secretary_average_last_one` we will select the last candidate.

```python
def secretary_average_zero(n, k, m):
    """
    Returns average value of chosen candidate.
    If no candidate is better than the best from the first group then count as zero.
    n - number of candidates
    m - number of random stets
    k - percent of candidates in test group
    """

    values = []
    k_index = ceil(n*k)
    for _ in range(m):
        candidates = np.random.rand(n)
        candidates_A = candidates[:k_index]
        candidates_B = candidates[k_index:]

        best_in_first_group = np.amax(candidates_A)

        for i in candidates_B:
            if i > best_in_first_group:
                values.append(i)
                break
        else:
            values.append(0)

    return np.average(values)

def secretary_average_last_one(n, k, m):
    """
    Returns average value of chosen candidate.
    If no candidate is better than the best from the first group then select last one.
    n - number of candidates
    m - number of random stets
    k - percent of candidates in test group
    """

    values = []
    k_index = ceil(n*k)
    for _ in range(m):
        candidates = np.random.rand(n)
        candidates_A = candidates[:k_index]
        candidates_B = candidates[k_index:]

        best_in_first_group = np.amax(candidates_A)

        for i in candidates_B:
            if i > best_in_first_group:
                values.append(i)
                break
        else:
            values.append(candidates_B[-1])

    return np.average(values)
```

For \\(n=100\\) and \\(m=10^6\\) we got a following result:

<div style="text-align:center">
<img src="/assets/sekretarki/2.png" alt="secretary" width="600"/>
</div>

What is interesting is that we no longer see maximum for the \\(1/e\\). Instead, we obtain it for \\(k \approx 8 \% \\).
Since the `secretary_average_last_one` gives better results we will focus on it in the following calculations.

# Is the distribution important?
Usually, in real life, most of the characteristic describing some person obeys the normal distribution.
In previous calculations we ware using flat distribution `candidates = np.random.rand(n)`.
Now lets change it to a normal distribution with \\(\mu=0.5\\) and \\(\sigma=1/6\\).
However, first we should assure that values in new set are fro the range \\([0,1]\\).
we can simply use `filter(lambda i: i>=0 and i<=1, set)` or some other function like:
```python
def get_normal(n, sigma):
    x = []
    while len(x) < n:
        y = np.random.normal(loc=0.5, scale=sigma, size=n)
        y = list(filter(lambda i: i>=0 and i<=1, y))
        x += y
    
    return x[:n]
```
to assure that the set has exactly \\(n\\) elements.

<div style="text-align:center">
<img src="/assets/sekretarki/3.png" alt="secretary" width="600"/>
</div>

We see that for a normal distribution of relative scores we got even worst results.
It is worth mentioning that in the original approach the maximal value of \\(P(k)\\) does not change for normal distribution.

# How to run these calculations faster?

Since we are interested in obtaining results for a variety of \\(k\\) we can use simple multiprocessing like:
```python
import concurrent.futures
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 50)[1:-1]

with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    f = partial(secretary_average_last_one_gauss, sigma=1/6, n=n, m=m)
    result = list(executor.map(f, x))
plt.plot(x, result, lw=1.5, label=r"Normal distribution $\sigma = 1/6$")
```
This will definitely speed up the calculations.

# Conclusions

If you hoped to get some real-life situation advice that is based on presented results I must disappoint you :P
However, it seams to be clear that the optimal value of \\(k\\) is between \\(8\%\\) and \\(37\%\\).

If you found any mistakes in this post please contact me.