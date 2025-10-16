import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional
from scipy.optimize import brentq


def CRR_tree_option_price_fast(
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
    Fast CRR option pricing - only returns price at root, no full tree storage.
    
    Returns:
        float: Option price at t=0
    """
    dt = ttm / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - y) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Only store current and next time step values (not entire tree)
    option_values = np.zeros(N + 1)
    
    # Terminal payoffs
    for j in range(N + 1):
        ST = S0 * (u ** (N - j)) * (d ** j)
        if option_type.upper() == 'C':
            option_values[j] = max(ST - K, 0)
        else:
            option_values[j] = max(K - ST, 0)
    
    # Backward induction - overwrite array in place
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = discount * (p * option_values[j] + (1 - p) * option_values[j + 1])
    
    return option_values[0]

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
    """Calculate implied volatility using fast CRR pricing."""
    def objective_func(sigma):
        return CRR_tree_option_price_fast(S0, K, r, y, ttm, sigma, N, option_type) - market_price
    
    try:
        return brentq(objective_func, 1e-6, 10.0, maxiter=1000)
    except:
        return np.nan


def Wrapper_implied_vol_crr_fast(row):
    """Wrapper function that can be imported by child processes. for use in multiprocessing."""
    return implied_vol_crr_fast(
        row['stock_exdiv'], 
        row['strike'], 
        row['risk_free'], 
        row['convenience_yield'],
        row['YTM'], 
        row['option_price'], 
        min(max(5 * int(row['DTM']), 1),500),  # Cap N to avoid excessive computation time
        row['cp_flag']
    )






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
