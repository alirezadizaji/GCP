import jax.numpy as jnp
import jax
import jaxopt
from jax import jit

import opt_einsum as oe


@jit
def normal(m, x):
    return (x - m) ** 2

@jit
def gamma(m, x, eps=1e-10):
    return x / (m + eps) + jnp.log(m + eps)

@jit
def rayleigh(m, x, eps=1e-10):
    return 2 * jnp.log(m + eps) + (jnp.pi/4) * (x / (m + eps)) ** 2

@jit
def weibull(m, x, eps=1e-10):
    return jnp.log(m + eps) + (m + eps) * jnp.log(x + eps) - (x / (m + eps)) ** (m + eps)

@jit
def poison_linear(m, x, eps=1e-10):
    return m - x * jnp.log(m + eps)

@jit
def poisson_log(m, x):
    return jnp.exp(m) - x * m

@jit
def bernoulli_odds(m, x, eps=1e-10):
    return jnp.log(m + 1) - x * jnp.log(m + eps)

@jit
def bernoulli_logit(m, x):
    return jnp.log(1 + jnp.exp(m)) - x * m

@jit
def negative_binomial(m, x, r, eps=1e-10):
    return (r + x) * jnp.log(1 + m) - x * jnp.log(m + eps)


# Default loss function
loss_fun = normal

