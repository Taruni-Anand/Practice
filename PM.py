"""Perform either exact or approximate inference to obtain answers to part III of Assignment 4.
You solved this inference problem exactly, and the answers should be P(G1=2|X2=50) = 0.1054 and p(X3=50|X2=50) = 0.1024.
If you're going to use Edward, I wasn't able to get any of the sampling-based inference procedures
(Metropolis-Hastings, Gibbs, hybrid Monte Carlo) to work on discrete RVs; however KLpq does seem to get a solution,
as long as you include the argument n_samples=100 or larger. Because there aren't any good examples of discrete RVs in
Edward, we found this implementation of the sprinkler/rain graphical model to be helpful. Read the description of
KLpq carefully: it does a search over Gaussian RVs, so you need to constrain the variable if you want it to be
nonnegative or binary. We also found that for estimating p(X3=50|X2=50), the distribution needs to be initialized to
be in the right neighborhood.

If you get really stuck and can't get this example to run,
implement the burglar alarm network from class and show some inference results.
The burglar alarm should be a straightforward extension of the sprinkler/rain net.
 We will give a max of 80% credit for this model.

As I mentioned on Piazza, one student has had success with PyMC3 and the code produced was quite sensible and readable.

For Part II, we would like you to hand in your code, and the runs that produce the two answers"""

import tensorflow as tf
print("tensorflow version: %s" % tf.__version__)
import edward as ed
print("edward version: %s" % ed.__version__)
import edward.models as edm

import edward.inferences as edi

matplotlib inline
config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

rain = edm.Bernoulli(probs=0.2)
p_sprinkler = tf.where(tf.cast(rain, tf.bool), 0.01, 0.4)
sprinkler = edm.Bernoulli(probs=p_sprinkler)
p_grass_wet = tf.where(tf.cast(rain, tf.bool),
                       tf.where(tf.cast(sprinkler, tf.bool), 0.99, 0.8),
                       tf.where(tf.cast(sprinkler, tf.bool), 0.9, 0.00000001))
grass_wet = edm.Bernoulli(probs=p_grass_wet)

with tf.Session():
    plt.hist([grass_wet.eval() for _ in range(1000)]);

q_rain = edm.Bernoulli(probs=tf.nn.sigmoid(tf.Variable(tf.random_normal([]))))
ed.get_session()
inf = edi.KLpq({rain: q_rain}, data={grass_wet: tf.constant(1, dtype=tf.int32)})
inf.run(n_samples=50)
print(q_rain.probs.eval())