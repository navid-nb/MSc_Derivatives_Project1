"""Implementation of the CRR Tree"""

from math import exp, sqrt
from typing import Tuple, Literal, Optional, List
from scipy.optimize import minimize_scalar


def crr_params(r: float,
               q: float,
               sigma: float,
               dt: float) -> Tuple[float, float, float, float]:
    """Compute Cox-Ross-Rubinstein tree parameters.

    Args:
        r (float): Continuously compounded risk-free rate per year.
        q (float): Continuous dividend yield per year. Use 0.0 if none.
        sigma (float): Annualized volatility (standard deviation).
        dt (float): Time step in years (T / N).

    Returns:
        Tuple[float, float, float, float]:
            u (float): Up factor.
            d (float): Down factor.
            p (float): Risk-neutral up probability.
            disc (float): Per-step discount factor exp(-r*dt).

    Raises:
        ValueError: If dt <= 0, sigma < 0, or if p is not between 0 and 1.

    Notes:
        The CRR construction uses:
            u = exp(sigma * sqrt(dt))
            d = 1 / u
            p = (exp((r - q) * dt) - d) / (u - d)
            disc = exp(-r * dt)
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")

    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    growth = exp((r - q) * dt)
    denom = (u - d)
    if denom == 0.0:
        raise ValueError("u and d are equal; choose a different dt or sigma.")
    p = (growth - d) / denom
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "Risk-neutral probability is outside [0, 1]. "
            "Check inputs (r, q, sigma, dt)."
        )
    disc = exp(-r * dt)
    return u, d, p, disc


def crr_price_european(S0: float,
                       K: float,
                       r: float,
                       q: float,
                       sigma: float,
                       T: float,
                       N: int,
                       option_type: Literal["C", "P"] = "C") -> float:
    """Price a European option using the Cox-Ross-Rubinstein binomial tree.

    Args:
        S0 (float): Spot price at time 0.
        K (float): Strike price.
        r (float): Continuously compounded risk-free rate per year.
        q (float): Continuous dividend yield per year. Use 0.0 if none.
        sigma (float): Annualized volatility.
        T (float): Time to maturity in years.
        N (int): Number of time steps in the tree (depth).
        option_type (Literal["call", "put"], optional): Type of option. Defaults to "call".

    Returns:
        float: Option price at time 0.

    Raises:
        ValueError: If inputs are invalid.

    Examples:
        >>> price = crr_price_european(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, N=200, option_type="call")
        >>> round(price, 4)
        10.45
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if T <= 0.0:
        raise ValueError("T must be positive.")
    if K < 0.0 or S0 < 0.0:
        raise ValueError("S0 and K must be non-negative.")

    dt = T / N
    u, d, p, disc = crr_params(r, q, sigma, dt)

    # Terminal asset prices S_T(i) = S0 * u^i * d^(N-i) for i = 0..N
    # Compute payoffs at maturity
    ST_values = [S0 * (u ** i) * (d ** (N - i)) for i in range(N + 1)]
    if option_type == "C":
        values = [max(s - K, 0.0) for s in ST_values]
    elif option_type == "P":
        values = [max(K - s, 0.0) for s in ST_values]
    else:
        raise ValueError('option_type must be "C" or "P".')

    # Backward induction
    for step in range(N - 1, -1, -1):
        # At each node j in this step, take discounted expectation of the two child nodes
        # values[j] <- disc * (p * values[j+1] + (1-p) * values[j])
        for j in range(step + 1):
            continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            values[j] = continuation

    return values[0]


def crr_price_american(S0: float,
                       K: float,
                       r: float,
                       q: float,
                       sigma: float,
                       T: float,
                       N: int,
                       option_type: Literal["C", "P"] = "P") -> float:
    """Price an American option using the Cox-Ross-Rubinstein binomial tree.

    Args:
        S0 (float): Spot price at time 0.
        K (float): Strike price.
        r (float): Continuously compounded risk-free rate per year.
        q (float): Continuous dividend yield per year. Use 0.0 if none.
        sigma (float): Annualized volatility.
        T (float): Time to maturity in years.
        N (int): Number of time steps in the tree (depth).
        option_type (Literal["C", "P"], optional): Type of option. Defaults to "P".

    Returns:
        float: Option price at time 0 allowing early exercise.

    Raises:
        ValueError: If inputs are invalid.

    Notes:
        Early exercise is relevant mainly for puts, and for calls only when there are payouts such as dividends.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if T <= 0.0:
        raise ValueError("T must be positive.")
    if K < 0.0 or S0 < 0.0:
        raise ValueError("S0 and K must be non-negative.")

    dt = T / N
    u, d, p, disc = crr_params(r, q, sigma, dt)

    # Precompute stock prices layer-by-layer to avoid recomputation
    # Terminal prices
    ST_values = [S0 * (u ** i) * (d ** (N - i)) for i in range(N + 1)]
    if option_type == "C":
        values = [max(s - K, 0.0) for s in ST_values]
    elif option_type == "P":
        values = [max(K - s, 0.0) for s in ST_values]
    else:
        raise ValueError('option_type must be "C" or "P".')

    # Backward induction with early exercise
    for step in range(N - 1, -1, -1):
        # Stock prices at this step: S(step, j) = S0 * u^j * d^(step - j)
        S_step: List[float] = [S0 * (u ** j) * (d ** (step - j)) for j in range(step + 1)]
        for j in range(step + 1):
            continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if option_type == "C":
                exercise = max(S_step[j] - K, 0.0)
            else:
                exercise = max(K - S_step[j], 0.0)
            values[j] = max(exercise, continuation)

    return values[0]

def solve_imp_vol(
        option_type: str, 
        option_price: float,
        american_option: bool,
        S0: float,
        T: float,
        K: float,
        q: float,
        r: float,
        N: int          
        ) -> float:
    """ Calibrating the implied volatility using Golden Section Search

    Args:
        option_type (str): Type of the option, i.e. 'C' or 'P'
        option_price (float): Market price of the option
        american_option (bool): True if the option is American, False if European
        S0 (float): Spot price at time 0
        T (float): Time to maturity in years
        K (float): strike price
        q (float): Continuous dividend yield per year. Use 0.0 if none.
        r (float): Continuously compounded risk-free rate per year
        N (int): Number of time steps in the tree (depth)

    Returns:
        float: calibrated implied volatility
    """
    def objective_func(sigma):
        if american_option:
            model_price = crr_price_american(S0, K, r, q, sigma, T, N, option_type)
        else:
            model_price = crr_price_european(S0, K, r, q, sigma, T, N, option_type)
        return (model_price - option_price) ** 2
    
    res = minimize_scalar(objective_func, bounds=(0.01, 5.0), method="bounded", options={"xatol": 1e-6, "maxiter": 1000})
    sigma_hat = res.x
    return sigma_hat