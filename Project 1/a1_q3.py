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
    price_tree = np.full((N+1, N+1), np.nan)
    price_tree[0, 0] = S_0
    for i in range(1, N+1):
        price_tree[:i, i] = u * price_tree[:i, i-1]
        price_tree[i, i] = d * price_tree[i-1, i-1]
    
    # binomial tree for option value
    value_tree = np.full_like(price_tree, np.nan)
    if option_type == 'call':
        value_tree[:, -1] = np.maximum(0, price_tree[:, -1] - K)
    else:
        value_tree[:, -1] = np.maximum(0, K - price_tree[:, -1])
    for i in range(N, 0, -1):
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
    bank_tree = value_tree - delta_tree * price_tree
    return delta_tree, bank_tree


def exercise_boundary(value_tree, price_tree, K):
    payoff_tree = K - price_tree
    exercised = np.where(value_tree == payoff_tree, 1, 0)
    exercised_price = np.where(exercised == 1, price_tree, -1)
    return exercised_price.max(axis=0)


def sim(S_0, r, mu, sigma, N, T, N_paths):
    delta_t = T / N
    p = 0.5 * (1 + (mu - r - 0.5*sigma**2) * np.sqrt(delta_t) / sigma)
    paths = np.full((N_paths, N+1), np.nan)
    paths[:, 0] = S_0
    for i in range(N):
        U = np.random.rand(N_paths)
        epsilon = (+1)*(U < p) + (-1)*(U >= p)
        paths[:, i+1] = paths[:, i] * np.exp(r*delta_t + sigma*np.sqrt(delta_t)*epsilon)
    return paths


def generate_pnl(paths, boundary, V_0, K, r, N, T):
    exercised = np.where(paths <= boundary, 1, 0)
    exercised[:, -1] = 1
    exercised_time = np.argmax(exercised == 1, axis=1)
    pnl = []
    delta_t = T / N
    for i, n in enumerate(exercised_time):
        pnl.append(np.exp(-r * delta_t * n) * np.maximum(K - paths[i][n], 0) - V_0)
    return pnl


if __name__ == '__main__':
    np.random.seed(1928)
    
    S_0 = 10.0
    K = 10.0
    r = 0.02
    q = 0.0
    mu = 0.05
    sigma = 0.2
    N = 5_000
    T = 1.0
    option_type = 'put'
    exercise_type = 'american'
    drift = r
    N_paths = 10_000
    
    exact_price = BSMOptionPricer(S_0, K, r, q, sigma, T, option_type)
    value_tree, price_tree = CRROptionPricer(S_0, K, r, q, sigma, N, T, option_type, exercise_type, drift)
    delta_tree, bank_tree = delta_hedging(value_tree, price_tree)
    
    print(exact_price)
    print(value_tree[0, 0], delta_tree[0, 0])
    
    boundary = exercise_boundary(value_tree, price_tree, K)
    boundary_for_plot = [x for x in boundary if x >= 0.0]
    ts = np.linspace(0, 1, N+1, endpoint=True)[-len(boundary_for_plot):]
    
    fig, ax = plt.subplots()
    ax.plot(ts, boundary_for_plot)
    fig.suptitle("American Put Option Exercise Boundary")
    plt.xlabel('time (t)')
    plt.ylabel('asset price (S)')
    plt.tight_layout()
    fig.savefig('exercise_boundary.png')
    # plt.show()
    
    fig, ax = plt.subplots()
    for t in (0, 1/4, 1/2, 3/4):
        ax.plot(price_tree[:, int(N*t/T)], delta_tree[:, int(N*t/T)], label=f"$t = {{{t}}}$")
    ax.plot(price_tree[:, int(N/T)], np.where(price_tree[:, int(N/T)] <= 10, -1, 0), label=f"$t = 1$")  # t = 1
    fig.suptitle("American Put Option Hedging Strategy - Risky Asset")
    plt.xlabel('spot price (S)')
    plt.ylabel('number of shares')
    ax.set_xlim(left=0.0, right=20.0)
    ax.legend()
    plt.tight_layout()
    fig.savefig('hedging_strategy_risky_asset.png')
    # plt.show()
    
    fig, ax = plt.subplots()
    for t in (0, 1/4, 1/2, 3/4):
        ax.plot(price_tree[:, int(N*t/T)], bank_tree[:, int(N*t/T)], label=f"$t = {{{t}}}$")
    ax.plot(price_tree[:, int(N/T)], np.where(price_tree[:, int(N/T)] <= 10, S_0, 0), label=f"$t = 1$")  # t = 1
    fig.suptitle("American Put Option Hedging Strategy - Bank Account")
    plt.xlabel('spot price (S)')
    plt.ylabel('dollar value ($)')
    ax.set_xlim(left=0.0, right=20.0)
    ax.legend()
    plt.tight_layout()
    fig.savefig('hedging_strategy_bank_account.png')
    # plt.show()
    
    sigma_list = [[0.10, 0.15], [0.25, 0.30]]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            value_tree, price_tree = CRROptionPricer(S_0, K, r, q, sigma_list[i][j], N, T, option_type, exercise_type, drift)
            delta_tree, bank_tree = delta_hedging(value_tree, price_tree)
            for t in (0, 1/4, 1/2, 3/4):
                axs[i, j].plot(price_tree[:, int(N*t/T)], delta_tree[:, int(N*t/T)], label=f"$t = {{{t}}}$")
            axs[i, j].plot(price_tree[:, int(N/T)], np.where(price_tree[:, int(N/T)] <= 10, -1, 0), label=f"$t = 1$")  # t = 1
            # axs[i, j].set(xlabel='spot price (S)', ylabel='number of shares')
            axs[i, j].set_xlim(left=0.0, right=20.0)
            axs[i, j].set_title(f"$\sigma = {{{sigma_list[i][j]}}}$")
            axs[i, j].legend(fontsize="x-small")
    # fig.suptitle(f"American Put Option Hedging Strategy for Various $\sigma$")
    plt.setp(axs[-1, :], xlabel='spot price (S)')
    plt.setp(axs[:, 0], ylabel='number of shares')
    plt.tight_layout()
    fig.savefig('hedging_strategy_risky_asset_sigmas.png')
    # plt.show()
    
    r_list = [[0.00, 0.01], [0.05, 0.10]]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            value_tree, price_tree = CRROptionPricer(S_0, K, r_list[i][j], q, sigma, N, T, option_type, exercise_type, drift)
            delta_tree, bank_tree = delta_hedging(value_tree, price_tree)
            for t in (0, 1/4, 1/2, 3/4):
                axs[i, j].plot(price_tree[:, int(N*t/T)], delta_tree[:, int(N*t/T)], label=f"$t = {{{t}}}$")
            axs[i, j].plot(price_tree[:, int(N/T)], np.where(price_tree[:, int(N/T)] <= 10, -1, 0), label=f"$t = 1$")  # t = 1
            # axs[i, j].set(xlabel='spot price (S)', ylabel='number of shares')
            axs[i, j].set_xlim(left=0.0, right=20.0)
            axs[i, j].set_title(f"$r = {{{r_list[i][j]}}}$")
            axs[i, j].legend(fontsize="x-small")
    # fig.suptitle(f"American Put Option Hedging Strategy for Various $r$")
    plt.setp(axs[-1, :], xlabel='spot price (S)')
    plt.setp(axs[:, 0], ylabel='number of shares')
    plt.tight_layout()
    fig.savefig('hedging_strategy_risky_asset_rs.png')
    # plt.show()
    
    paths = sim(S_0, r, mu, sigma, N, T, N_paths)
    fig, ax = plt.subplots()
    ax.plot(paths.T, c='b', alpha=1/255)
    plt.show()
    
    pnl = generate_pnl(paths, boundary, value_tree[0, 0], K, r, N, T)
    # print(np.sum(pnl) / N_paths)
    fig, ax = plt.subplots()
    ax.hist([x for x in pnl if x > -value_tree[0, 0]], bins=20)
    plt.show()
