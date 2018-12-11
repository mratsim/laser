Random sampling is a key part of probabilistic programming, bayesian deep learning and text generation.

For example for text generation with a 4-character model "abcd", the network might predict the next character probability with [0.1, 0.3, 0.4, 0.2] and we need to sample a character from this (multinomial) distribution.

Now imagine that we instead have a vocabulary of a hundred thousands of words to sample from and here is the need for scalable performance.

## Multinomial distribution - Weighted Random sampling

### Use case
This is a critical distribution to optimise and the one used to sample next words or characters in NLP text generation.
It is also used to give more weight to certain labels during training by sampling the training set with a skewed distribution (instead of scaling the gradient by a certain weight).
And it's probably also the best way to implement stratified K-Fold

### What is it
It is the generalisation of the binomial distribution where for example probability of a biaised coin toss could be [0.25, 0.75].

### Implementation

#### In research

Papers
  - [Random Sampling from Databases](http://db.cs.berkeley.edu/papers/UCB-PhD-olken.pdf), 1993, Olken
    - p22 goes over Acceptance/Rejection sampling, Partial Sum Trees (Wong et al), Alias Method

Let's start with the main ones:
  - Linear search
  - Binary search on top of the cumulative distribution (see http://www.keithschwarz.com/darts-dice-coins/)
  - [Alias method](https://en.wikipedia.org/wiki/Alias_method)
  - [Using Fenwick trees](https://www.cs.utexas.edu/~rofuyu/papers/nomad-lda-www.pdf) (A Scalable Asynchronous Distributed Algorithm for
Topic Modeling, state-of-the-art, scalable to billions of words and parallel)
    - Data structure largely inspired by [An Efficient Method for Weighted Sampling without Replacement](https://www.researchgate.net/profile/Malcolm_Easton/publication/220617264_An_Efficient_Method_for_Weighted_Sampling_Without_Replacement/links/0a85e53a492355b285000000/An-Efficient-Method-for-Weighted-Sampling-Without-Replacement.pdf), 1980, Wong et al
    - [Parallel Prefix Sum](https://en.wikipedia.org/wiki/Prefix_sum)
    - [Heaps for incremental computation](https://timvieira.github.io/blog/post/2016/11/21/heaps-for-incremental-computation/) and [code](https://gist.github.com/timvieira/da31b56436045a3122f5adf5aafec515)
        ```python
        import numpy as np
        from numpy.random import uniform


        def update(S, k, v):
            "Update value position `k` in time O(log n)."
            d = S.shape[0]
            i = d//2 + k
            S[i] = v
            while i > 0:
                i //= 2
                S[i] = S[2*i] + S[2*i + 1]


        def sumheap(w):
            "Create sumheap from weights `w`."
            n = w.shape[0]

            # About the datastructure: Bottom-most level of the heap has size n' =
            # next-power-of-two(n). The number of internal nodes in the tree is n'-1. We
            # have a dummy node at position zero to make indexing math simpler. So, we
            # allocate twice the size of the bottom level to fit internal nodes. Thus,
            # the overal data structure is <4*n in the worst case because the next power
            # of two <2n and then we have another factor of two for internal nodes.
            d = int(2**np.ceil(np.log2(n)))
            S = np.zeros(2*d)

            # O(n) version (faster than calling update n times => O(n log n))
            S[d:d+n] = w
            for i in reversed(range(1, d)):
                S[i] = S[2*i] + S[2*i + 1]

            return S


        def check(S, i):
            "Check heap invariant."
            d = S.shape[0]
            if i >= d//2:   # only checks internal nodes.
                return
            assert S[i] == S[2*i] + S[2*i + 1]
            check(S, 2*i)
            check(S, 2*i + 1)


        def dump(S):
            "Print heap for debugging."
            for i in range(int(np.ceil(np.log2(len(S))))):
                print 'depth', i, S[2**i:2**(i+1)]


        def sample(w, u):
            "Ordinary sampling method, O(n) to build heap, O(log n) per sample after that."
            c = w.cumsum()
            return c.searchsorted(u * c[-1])


        def hsample(S, u):
            "Sample from sumheap, O(log n) per sample."
            offset = S.shape[0]//2  # number of internal nodes.
            # random probe
            p = S[1] * u
            # Use binary search to find the index of the largest CDF (represented as a
            # heap) value that is less than a random probe.
            i = 1
            while i < offset:
                # Determine if the value is in the left or right subtree.
                i *= 2
                left = S[i]
                if p > left:
                    # Value is in right subtree. Subtract mass under left subtree.
                    p -= left
                    i += 1
            return i - offset


        def main():
            for n in np.random.choice(range(1, 100), size=10):
                print n
                w = np.round(uniform(0, 10, size=n), 1)
                S = sumheap(w)
                check(S, 1)
                for _ in range(100):
                    # test uses same random number because the methods should be identical up to ties.
                    u = uniform()
                    p1 = sample(w, u)
                    p2 = hsample(S, u)
                    assert p1 == p2
                    # change a random value in the weight array
                    c = np.random.randint(n)
                    v = uniform(10)
                    w[c] = v
                    update(S, c, v)
                    check(S, 1)


        if __name__ == '__main__':
            main()
        ```

And a few curious ones:
  - [roulette wheel selection](https://en.wikipedia.org/wiki/Fitness_proportionate_selection) and http://www.keithschwarz.com/darts-dice-coins/

#### Misc literature
- [ZenLDA: Large-Scale Topic Model Training on Distributed
Data-Parallel Platform](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8268736)

#### Numpy

For [`choice`](https://github.com/numpy/numpy/blob/9297bd786f88c339ff421f74be5cf5ebc5fe2e2b/numpy/random/mtrand/mtrand.pyx#L1032-L1211) Numpy uses binary search on top of the cumulative distribution

```python
    def choice(self, a, size=None, replace=True, p=None):
        """
        Parameters
        -----------
        a : 1-D array-like or int. If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)
        size : int or tuple of ints, optional. Output shape.
        replace : boolean, optional. Whether the sample is with or without replacement
        p : 1-D array-like, optional. The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.
        Returns
        --------
        samples : single item or ndarray
            The generated random samples
        Examples
        ---------
        Generate a uniform random sample from np.arange(5) of size 3:
        >>> np.random.choice(5, 3)
        array([0, 3, 4])
        >>> #This is equivalent to np.random.randint(0,5,3)
        Generate a non-uniform random sample from np.arange(5) of size 3:
        >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        array([3, 3, 0])
        """

        ...

        # Actual sampling
        if replace:
            if p is not None:
                cdf = p.cumsum()
                cdf /= cdf[-1]
                uniform_samples = self.random_sample(shape)
                idx = cdf.searchsorted(uniform_samples, side='right')
                idx = np.array(idx, copy=False) # searchsorted returns a scalar
            else:
                idx = self.randint(0, pop_size, size=shape)
        else:
            if size > pop_size:
                raise ValueError("Cannot take a larger sample than "
                                 "population when 'replace=False'")

            if p is not None:
                if np.count_nonzero(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(shape, dtype=np.int)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = self.rand(size - n_uniq)
                    if n_uniq > 0:
                        p[flat_found[0:n_uniq]] = 0
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    new = cdf.searchsorted(x, side='right')
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq:n_uniq + new.size] = new
                    n_uniq += new.size
                idx = found
            else:
                idx = self.permutation(pop_size)[:size]
                if shape is not None:
                    idx.shape = shape
```

Note there is also Numpy multinomial that uses repeated binomial sampling but that doesn't match our need: [Numpy multinomial](https://github.com/numpy/numpy/blob/9297bd786f88c339ff421f74be5cf5ebc5fe2e2b/numpy/random/mtrand/mtrand.pyx#L4541-L4652)

Implementation of [searchsorted](https://github.com/numpy/numpy/blob/c2372aff1ea85e8e1d64d8b5ced7e362fd0f4a5a/numpy/core/src/multiarray/item_selection.c#L1634-L1782) and [Numpy binsearch](https://github.com/numpy/numpy/blob/464f79eb1d05bf938d16b49da1c39a4e02506fa3/numpy/core/src/npysort/binsearch.c.src#L39-L82)

Equivalent [CUDA code](https://github.com/aliutkus/pytorch_searchsorted/blob/86dc056ba75883abf2f25e0907299fc24fadba4a/src/searchsorted_cuda_kernel.cu#L39-L129)


#### PyTorch

PyTorch uses either the [alias method](https://github.com/pytorch/pytorch/blob/b039a715ce4e9cca82ae3bf72cb84652957b2844/aten/src/TH/generic/THTensorRandom.cpp#L136-L246) or [CDF + binary search](https://github.com/pytorch/pytorch/blob/b039a715ce4e9cca82ae3bf72cb84652957b2844/aten/src/TH/generic/THTensorRandom.cpp#L247-L423)

#### Tensorflow

https://github.com/tensorflow/tensorflow/blob/125bf1dbb76c05bf5f88f14e77387ce35f986621/tensorflow/core/kernels/multinomial_op.cc

### Reservoir sampling - one-pass sampling over the data

It is also possible to do weighted sampling in one pass over a stream of unknown length
with a technique called reservoir sampling.
  - [Wikipedia](https://en.wikipedia.org/wiki/Reservoir_sampling)
  - [Cloudera article](https://blog.cloudera.com/blog/2013/04/hadoop-stratified-randosampling-algorithm/)
  - [Excellent article with an easy example to follow](https://gregable.com/2007/10/reservoir-sampling.html)

Papers:
  - [Random Sampling with a reservoir](https://www.cs.umd.edu/~samir/498/vitter.pdf), 1985, Vitter
  - [An efficient algorithm for sequentiam random sampling](https://www.researchgate.net/profile/Jeffrey_Vitter/publication/278627791_An_efficient_algorithm_for_sequential_random_sampling/links/5747b27508aef66a78b08012/An-efficient-algorithm-for-sequential-random-sampling.pdf), 1987, Vitter
    - Introduces gap sampling
  - [Weighted Random Sampling](https://pdfs.semanticscholar.org/6798/85127be7c2b6fd0bb0de40ea148629235ccc.pdf), 2005, Efraimidis
  - [Weighted Random Sampling over Data streams](https://arxiv.org/pdf/1012.0256.pdf), 2010, Efraimidis
  - [Weighted random sampling without replacement from data streams](https://arxiv.org/pdf/1506.01747), 2016, Braverman
  - [Accelerating Weighted Random Sampling without replacement](https://www.ethz.ch/content/dam/ethz/special-interest/baug/ivt/ivt-dam/vpl/reports/1101-1200/ab1141.pdf), 2016, Muller

Articles:
  - [Fast reservoir sampling with gap sampling](http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/)


## Normal distribution

TODO

- [Box-Muller transform](https://en.wikipedia.org/wiki/Box–Muller_transform)
- [Polar method](https://en.wikipedia.org/wiki/Marsaglia_polar_method)
- [Ziggurat algorithm](https://en.wikipedia.org/wiki/Ziggurat_algorithm)

## Importance sampling for deep learning

- https://github.com/idiap/importance-sampling
- Not all samples are created equal: https://arxiv.org/abs/1803.00942
- Baised importance sampling: https://arxiv.org/abs/1706.00043
