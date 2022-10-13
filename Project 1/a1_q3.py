import numpy as np
import matplotlib.pyplot as plt
from statistics import NormalDist


def BSMOptionPricer(S_0, K, r, q, sigma, T, option_type):
    """Black-Scholes-Merton European option pricer

    Args:
        S_0 (float): current asset price
        K (float): option strike price
        r (float): annualized continuously compounding risk-free interest rate
        q (float): annualized continuously compounding dividend yield
        sigma (float): annualized asset price volatility
        T (float): time to maturity in years
        option_type (str): 'call' or 'put'

    Returns:
        _type_: _description_
    """

    d_1 = (1 / sigma * np.sqrt(T)) * (np.log(S_0 / K) + (r - q + sigma*sigma / 2) * T)
    d_2 = d_1 - sigma * np.sqrt(T)

    N_d_1 = NormalDist().cdf(d_1)
    N_d_2 = NormalDist().cdf(d_2)

    n_d_1 = NormalDist().pdf(d_1)

    q_discount_rate = np.exp(-q * T)
    r_discount_rate = np.exp(-r * T)

    if option_type == 'call':
        price = N_d_1 * S_0 * q_discount_rate - N_d_2 * K * r_discount_rate
        delta = q_discount_rate * N_d_1
        theta = (-S_0 * q_discount_rate * n_d_1 * sigma / (2 * np.sqrt(T))) - r * K * r_discount_rate * N_d_2 - q * S_0 * q_discount_rate * N_d_1
    else:
        price = (N_d_1 - 1) * S_0 * q_discount_rate - (N_d_2 - 1) * K * r_discount_rate
        delta = q_discount_rate * (N_d_1 - 1)
        theta = (-S_0 * q_discount_rate * n_d_1 * sigma / (2 * np.sqrt(T))) - r * K * r_discount_rate * (N_d_2 - 1) - q * S_0 * q_discount_rate * (N_d_1 - 1)

    gamma = q_discount_rate * n_d_1 / (S_0 * sigma * np.sqrt(T))

    return price, delta, gamma, theta / 365.0


def CRROptionPricer(S_0, K, r, q, sigma, N, T, option_type, exercise_type, drift=0.0):
    """Cox-Ross-Rubinstein (with drift) binomial tree option pricing model

    Args:
        S_0 (float): current asset price
        K (float): option strike price
        r (float): annualized continuously compounding risk-free interest rate
        q (float): annualized continuously compounding dividend yield
        sigma (float): annualized asset price volatility
        N (int): number of steps i.e., depth of the binomial tree
        T (float): time to maturity in years
        option_type (str): 'call' or 'put'
        exercise_type (str): 'european' or 'american'
        drift (float, optional): drift value in up & down factors. Defaults to 0.0.
    
    Returns:
        _type_: _description_
    """
    
    if option_type not in ('call', 'put'):
        raise ValueError("the argument 'option_type' must be either 'call' or 'put'")
    if exercise_type not in ('european', 'american'):
        raise ValueError("the argument 'exercise_type' must be either 'european' or 'american'")
    
    # useful constants to define
    delta_t = T / N
    discount_rate = np.exp(-r * delta_t)
    u = np.exp(drift*delta_t + sigma*np.sqrt(delta_t))
    d = np.exp(drift*delta_t - sigma*np.sqrt(delta_t))
    q_u = (np.exp((r - q) * delta_t) - d) / (u - d)
    q_d = 1 - q_u
    
    # binomial tree for asset price
    price_tree = np.full((N, N), np.nan)
    price_tree[0, 0] = S_0
    for i in range(1, N):
        price_tree[:i, i] = u * price_tree[:i, i-1]
        price_tree[i, i] = d * price_tree[i-1, i-1]
    
    # binomial tree for option value
    value_tree = np.full_like(price_tree, np.nan)
    if option_type == 'call':
        value_tree[:, -1] = np.maximum(0, price_tree[:, -1] - K)
    else:
        value_tree[:, -1] = np.maximum(0, K - price_tree[:, -1])
    for i in range(N-1, 0, -1):
        value_tree[:i, i-1] = discount_rate * (q_u * value_tree[:i, i] + q_d * value_tree[1:i+1, i])
        if exercise_type == 'american':
            if option_type == 'call':
                value_tree[:i, i-1] = np.maximum(price_tree[:i, i-1] - K, value_tree[:i, i-1])
            else:
                value_tree[:i, i-1] = np.maximum(K - price_tree[:i, i-1], value_tree[:i, i-1])
    
    return value_tree, price_tree


def delta_hedging(value_tree, price_tree):
    delta_tree = np.full_like(value_tree, np.nan)
    for i in range(delta_tree.shape[1]-1):
        delta_tree[:i+1, i] = (value_tree[:i+1, i+1] - value_tree[1:i+2, i+1]) / (price_tree[:i+1, i+1] - price_tree[1:i+2, i+1])
    return delta_tree


def exercise_boundary(value_tree, price_tree, K):
    payoff_tree = K - price_tree
    exercised = np.where(value_tree == payoff_tree, 1, 0)
    exercised_price = np.where(exercised == 1, price_tree, -1)
    return exercised_price.max(axis=0)


if __name__ == '__main__':    
    S_0 = 10.0
    K = 10.0
    r = 0.02
    q = 0.0
    sigma = 0.2
    N = 5000
    T = 1.0
    option_type = 'put'
    exercise_type = 'american'
    drift = r
    
    exact_price = BSMOptionPricer(S_0, K, r, q, sigma, T, option_type)
    value_tree, price_tree = CRROptionPricer(S_0, K, r, q, sigma, N, T, option_type, exercise_type, drift)
    delta_tree = delta_hedging(value_tree, price_tree)
    
    print(exact_price)
    print(value_tree[0, 0], delta_tree[0, 0])
    
    boundary = exercise_boundary(value_tree, price_tree, K)
    boundary_for_plot = [x for x in boundary if x >= 0.0]
    ts = np.linspace(0, 1, N, endpoint=True)[-len(boundary_for_plot):]
    
    fig, ax = plt.subplots()
    ax.plot(ts, boundary_for_plot)
    fig.suptitle("American Put Option Exercise Boundary")
    plt.xlabel('time (t)')
    plt.ylabel('asset price (S)')
    plt.tight_layout()
    # ax.set_ylim(bottom=0.0)
    # fig.savefig('exercise_boundary.png')
    plt.show()
    
    fig, ax = plt.subplots()
    for t in (0, 1/4, 1/2, 3/4):
        ax.plot(price_tree[:, int(N*t/T)], delta_tree[:, int(N*t/T)], label=f"$t = {{{t}}}$")
    fig.suptitle("American Put Option Hedging Strategy")
    plt.xlabel('spot price (S)')
    plt.ylabel('number of units of asset')
    ax.set_xlim(left=0.0, right=20.0)
    ax.legend()
    plt.tight_layout()
    # fig.savefig('hedging_strategy.png')
    plt.show()