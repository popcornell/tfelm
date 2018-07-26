# TF-ELM
**An experimental API for Extreme Learning machines Neural Networks made with TensorFlow.**

Extreme Learning Machines are a particular machine learning paradigm
based on random weights and biases.

This class of machine-learning techniques because are based upon random projections,
are trainable with almost negligible hardware and time resources.

In some contexts, their performance can be comparable to classical Multi Layer Perceptron Networks with the advantage of mantaining negligible training resources.

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
- [ML_ELM class](https://github.com/popcornell/tfelm/blob/pop_new/ML_ELM_class_example.ipynb)

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

#### Authors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
|[<img src="https://avatars0.githubusercontent.com/u/18726713?s=460&v=4https://avatars0.githubusercontent.com/u/18726713?s=460&v=4" height='150px' width="100px;"/><br /><sub><b>Samuele_Cornell</b></sub>]
 [<img src="https://avatars0.githubusercontent.com/u/36079710?s=460&v=4" height='150px' width="100px;"/><br /><sub><b>Andrea Castellani</b></sub>]

<!-- ALL-CONTRIBUTORS-LIST:END -->
