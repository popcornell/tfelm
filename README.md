# TF-ELM
**An experimental API for Extreme Learning machines Neural Networks made with TensorFlow.**

Extreme Learning Machines are a particular neural-network and learning paradigm
based on random weights and biases.

It seems that these Networks are also a controversial topic in Literature due to the fact
that for some researchers are strongly based, or even a rip-off, of prior works such as
random Projections.

Academic allegations aside, these learning methods due to the fact that are trainable with
almost negligible hardware and time resources are worth investigating.

In fact, in some contexts, their performance can be comparable to classical Multi Layer Perceptron Networks with the advantage of mantaining negligible training resources.

This makes these networks ideally suited for fast-prototyping and for certain big-data problems where a result should be obtained in a
reasonable time and/or computing resources for more training-intensive but better models aren’t
available.



#### Support for:

- Single Hidden layer ELM.
- Multi layer ELM.
- Custom weights initialization.
- Custom activation functions.
- TensorFlow Dataset API.
- Full multi-GPU support and model serving through TensorFlow.

#### Examples available through Jupyter Notebooks:
- [ELM class](https://github.com/popcornell/tfelm/blob/pop_new/ELM_class_example.ipynb)
- ML_ELM class # TODO


#### More on ELMs vs MLPs

While ELMs require a training time which is order of magnitude smaller than a performance-wise comparable MLP,
they usually require more hidden neurons than the MLP network to reach the same performance.

Because of this it can be argued that Single layer ELMs and MLPs are based on a somewhat antithetical philosophy, ELMs bet on
a large number of hidden layer units, MLPs on the other hand bet on a small but properly trained
number of hidden units.
This leads to more training time for MLPs but on the other hand more compact networks which
are less computationally intensive in the feedfoward phase.

ELMs, on the other hand, have trivial training time but produce larger networks which place more
computational burden on platforms when they become part of an actual application.
