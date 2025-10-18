import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional
from scipy.optimize import brentq


def CRR_tree_option_price_american_fast(
    S0: float,
    K: float,
    r: float,
    y: float,
    ttm: float,
    sigma: float,
    N: int,
    option_type: str = 'C'
) -> float:
    """
    Price an American option using the Cox-Ross-Rubinstein (CRR) binomial tree model.
    This implementation is fully vectorized for speed and memory efficiency.

    Parameters:
        S0 (float): Initial stock price (ex-dividend).
        K (float): Strike price of the option.
        r (float): Risk-free interest rate (annual, continuously compounded).
        y (float): Continuous dividend yield (annual).
        ttm (float): Time to maturity in years.
        sigma (float): Volatility (annualized standard deviation).
        N (int): Number of binomial steps in the tree.
        option_type (str): 'C' for call, 'P' for put.

    Returns:
        float: The price of the American option at time t=0.

    Notes:
        - Uses backward induction and checks for early exercise at each node.
        - Vectorized for high performance with large N.
        - Raises ValueError for invalid parameters.
    """
    # Validate input parameters
    if ttm / N <= 0.0:
        raise ValueError("dt must be positive.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")

    # Calculate binomial model parameters
    dt = ttm / N
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u                       # Down factor

    if abs(u - d) <= 1e-10:
        raise ValueError("u-d ~= 0")

    p = (np.exp((r - y) * dt) - d) / (u - d)  # Risk-neutral up probability
    discount = np.exp(-r * dt)                # Discount factor per step

    if not (0.0 <= p <= 1.0):
        raise ValueError("Risk-neutral probability is not between 0 and 1")

    # Pre-compute powers for all steps (vectorized for speed)
    j = np.arange(N + 1)
    u_powers = u ** j
    d_powers = d ** j

    # Compute terminal stock prices and option payoffs at maturity
    # ST = S0 * u_powers * d_powers[N] / d_powers
    ST = S0 * u_powers * d_powers[::-1]

    if option_type.upper() == "C":
        values = np.maximum(ST - K, 0.0)
    elif option_type.upper() == "P":
        values = np.maximum(K - ST, 0.0)
    else:
        raise ValueError('option_type must be "C" or "P".')

    # Backward induction: step back through the tree, checking for early exercise
    for step in range(N - 1, -1, -1):
        # Stock prices at this step using pre-computed powers
        # S_step = S0 * u_powers[:step + 1] * d_powers[step] / d_powers[:step + 1]
        S_step = S0 * u_powers[:step + 1] * d_powers[step::-1]

        # Calculate continuation and early exercise values
        continuation = discount * (p * values[1:step + 2] + (1.0 - p) * values[:step + 1])
        if option_type.upper() == "C":
            exercise = np.maximum(S_step - K, 0.0)
        else:
            exercise = np.maximum(K - S_step, 0.0)

        # American option: take the maximum of exercise and continuation value
        values[:step + 1] = np.maximum(exercise, continuation)

    return values[0]


def implied_vol_crr_fast(
    S0: float,
    K: float,
    r: float,
    y: float,
    ttm: float,
    market_price: float,
    N: int,
    option_type: str = 'C'
) -> float:
    """
    Calculate the implied volatility of an option using the fast CRR binomial tree.

    Parameters:
        S0 (float): Initial stock price (ex-dividend).
        K (float): Strike price.
        r (float): Risk-free interest rate (annual, continuously compounded).
        y (float): Continuous dividend yield (annual).
        ttm (float): Time to maturity in years.
        market_price (float): Observed market price of the option.
        N (int): Number of binomial steps in the tree.
        option_type (str): 'C' for call, 'P' for put.

    Returns:
        float: Implied volatility (annualized) that matches the market price.

    Notes:
        - Uses Brent's method (brentq) to find the root of the price difference.
        - Returns the volatility that makes the model price equal to the market price.
        - Raises ValueError if no root is found in the interval.
    """
    def objective_func(sigma):
        # Difference between model price and market price for a given sigma
        return CRR_tree_option_price_american_fast(S0, K, r, y, ttm, sigma, N, option_type) - market_price

    # Find the root (implied vol) in the interval [0.05, 3.0]
    return brentq(objective_func, 0.05, 3.0, maxiter=1000)


def Wrapper_implied_vol_crr_fast(row):
    """
    Wrapper for multiprocessing: computes implied volatility for a single DataFrame row.

    Parameters:
        row (pd.Series): A row from a DataFrame containing all required option parameters.

    Returns:
        float: Implied volatility for the option described by the row.

    Notes:
        - Designed for use with joblib or other parallel processing tools.
        - Calls implied_vol_crr_fast with parameters extracted from the row.
    """
    return implied_vol_crr_fast(
        row['stock_exdiv'], 
        row['strike'], 
        row['risk_free'], 
        row['convenience_yield'],
        row['YTM'], 
        row['option_price'], 
        max(5 * int(row['DTM']), 1),
        # min(max(5 * int(row['DTM']), 1), 500),  # Cap N to avoid excessive computation time
        row['cp_flag']
    )






#==============================================================================    
#just for fun(not in the assignment): a function for outputing the whole tree and plotting it
#==============================================================================   

  
def CRR_tree_option_price(
    S0: float,
    K: float,
    r: float,
    y: float,
    ttm: float,
    sigma: float,
    N: int,
    option_type: str = 'C'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate option values and underlying stock prices on all nodes 
    using a CRR binomial tree with continuous dividend yield.

    Args:
        S0 (float): Initial stock price (ex-dividend).
        K (float): Strike price.
        r (float): Risk-free interest rate (continuous compounding).
        y (float): Continuous dividend yield.
        ttm (float): Time to maturity in years.
        sigma (float): Volatility.
        N (int): Number of binomial steps.
        option_type (str): 'C' for call, 'P' for put.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            option_values: 2D matrix (N+1 x N+1) of option values (unused entries zero)
            stock_prices: 2D matrix (N+1 x N+1) of stock prices (unused entries zero)
    """
    dt = ttm / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - y) * dt) - d) / (u - d)

    # Build stock price tree (forward induction)
    stock_prices = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Set values at maturity (last column)
    option_values = np.zeros((N + 1, N + 1))
    ST = stock_prices[:N + 1, N]
    if option_type.upper() == 'C':
        option_values[:N + 1, N] = np.maximum(ST - K, 0)
    elif option_type.upper() == 'P':
        option_values[:N + 1, N] = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'C' or 'P'")

    # Backward induction for option value tree
    discount = np.exp(-r * dt)
    for i in reversed(range(N)):
        for j in range(i + 1):
            option_values[j, i] = discount * (
                p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1]
            )

    return option_values, stock_prices

def implied_vol_crr(
    S0: float,
    K: float,
    r: float,
    y: float,
    ttm: float,
    market_price: float,
    N: int,
    option_type: str = 'C'
) -> float:
    """
    Calculate implied volatility by inverting CRR binomial option price.

    Args:
        S0 (float): Stock price ex-dividend.
        K (float): Strike price.
        r (float): Risk-free rate.
        y (float): Continuous dividend yield.
        ttm (float): Time to maturity in years.
        market_price (float): Observed option market price.
        N (int): Number of binomial steps.
        option_type (str): 'C' for call, 'P' for put.

    Returns:
        float: Implied volatility or NaN if no root found.
    """
    def objective_func(sigma):
        option_values, _ = CRR_tree_option_price(S0, K, r, y, ttm, sigma, N, option_type)
        return option_values[0, 0] - market_price

    return brentq(objective_func,
                           1e-6,  # lower bound
                           10.0,  # upper bound
                           maxiter=1000)

def plot_binomial_tree(
    option_matrix: Optional[np.ndarray] = None,
    stock_matrix: Optional[np.ndarray] = None,
    title: str = "Binomial Tree - Stock Price & Option Value"
) -> None:
    """
    Plot binomial tree graph with node labels showing both underlying stock prices and option values.
    Either or both of option_matrix and stock_matrix can be provided.

    Args:
        option_matrix (np.ndarray, optional): 2D array of option values at nodes (N+1 by N+1).
        stock_matrix (np.ndarray, optional): 2D array of underlying stock prices at nodes (N+1 by N+1).
        title (str): Plot title, will dynamically adapt if only one input is given.

    Returns:
        None
    """
    if option_matrix is None and stock_matrix is None:
        print("No input matrices given: plot will be empty.")
        plt.figure()
        plt.title(title + " (Empty)")
        plt.show()
        return

    if option_matrix is not None:
        N = option_matrix.shape[1] - 1
    elif stock_matrix is not None:
        N = stock_matrix.shape[1] - 1
    else:
        return

    G = nx.DiGraph()
    pos = {}
    labels = {}

    for time_step in range(N + 1):
        for node_index in range(time_step + 1):
            node = (node_index, time_step)
            G.add_node(node)
            pos[node] = (time_step, -node_index)
            label_parts = []
            if stock_matrix is not None:
                S_price = stock_matrix[node_index, time_step]
                label_parts.append(f"S: {S_price:.2f}")
            if option_matrix is not None:
                V_price = option_matrix[node_index, time_step]
                label_parts.append(f"V: {V_price:.2f}")

            labels[node] = '\n'.join(label_parts)

    for time_step in range(N):
        for node_index in range(time_step + 1):
            G.add_edge((node_index, time_step), (node_index, time_step + 1))
            G.add_edge((node_index, time_step), (node_index + 1, time_step + 1))

    plt.figure(figsize=(14, 8))
    nx.draw_networkx_edges(G, pos, arrows=False)
    # Draw nodes only if needed (size=0 to remove dots)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')

    # Draw labels with red, bold font
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=9,
        font_color='red',
        font_weight='bold'
    )
    
    dynamic_title = title
    if option_matrix is None and stock_matrix is not None:
        dynamic_title = "Binomial Tree - Stock Price Only"
    elif stock_matrix is None and option_matrix is not None:
        dynamic_title = "Binomial Tree - Option Value Only"

    plt.title(dynamic_title)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
