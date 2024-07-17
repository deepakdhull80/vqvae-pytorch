# Vector Quantized Variational Autoencoder

## 1. Variational Autoencoder

A Variational Autoencoder (VAE) is a type of generative model that combines neural networks and probabilistic modeling to learn a latent representation of input data and generate new, similar data.

#### Encoder:

The encoder maps input data x to a latent space.
Instead of mapping x to a single point in the latent space, it maps x to a distribution, typically a Gaussian with mean μ(x) and standard deviation σ(x).
#### Latent Space:

The latent space is where the compressed representation of the input data resides.
A random sample 𝑧 is drawn from the latent distribution N(μ(x),σ(x)).

#### Decoder:
The decoder maps the sampled latent variable z back to the data space to reconstruct the input data X.
It generates data that resembles the input distribution from the latent variable.
#### Loss Function:

The VAE uses a loss function that combines two terms:
1. Reconstruction Loss: Measures how well the output of the decoder matches the input data. This can be done using Mean Squared Error (MSE) or Binary Cross-Entropy (BCE).
2. KL Divergence: Ensures that the learned latent space distribution is close to a prior distribution (typically a standard normal distribution). It acts as a regularizer to prevent overfitting and ensures a well-structured latent space.

#### References
1. Understanding Variational Autoencoders [Link](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
