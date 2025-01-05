import numpy as np
from itertools import product


def price_multi_asset_bermudan_max_call(
    S0_vector,             # Initial stock prices
    K=100,                 # Strike price
    T=3,                   # Time to maturity
    r=0.05,               # Risk-free rate
    q_vector=None,         # Dividend yields
    sigma_vector=None,     # Volatilities
    rho_matrix=None,       # Correlation matrix
    M=9,                   # Number of exercise opportunities
    num_steps=50          # Number of time steps per period
):
    """
    Price a multi-asset Bermudan max call option for both 2 and 3 asset cases
    """
    num_assets = len(S0_vector)
    if num_assets not in [2, 3]:
        raise ValueError("This implementation supports only 2 or 3 assets")
        
    # Set default values if not provided
    if q_vector is None:
        q_vector = [0.1] * num_assets
    if sigma_vector is None:
        sigma_vector = [0.2] * num_assets
    if rho_matrix is None:
        rho_matrix = np.eye(num_assets)
        
    dt = T/num_steps
    
    # Convert inputs to numpy arrays
    S0_vector = np.array(S0_vector)
    q_vector = np.array(q_vector)
    sigma_vector = np.array(sigma_vector)
    rho_matrix = np.array(rho_matrix)
    
    # Calculate up and down factors for each asset
    u = np.array([np.exp(sigma * np.sqrt(dt)) for sigma in sigma_vector])
    d = np.array([1/up_factor for up_factor in u])
    
    # Calculate risk-neutral probabilities for each asset
    p = (np.exp((r - q_vector) * dt) - d)/(u - d)
    
    def get_transition_probs_2d():
        """Calculate transition probabilities for 2-asset case"""
        p_matrix = np.zeros(4)
        p_matrix[0] = p[0]*p[1] + rho_matrix[0,1]*np.sqrt(p[0]*(1-p[0])*p[1]*(1-p[1]))  # uu
        p_matrix[1] = p[0]*(1-p[1]) - rho_matrix[0,1]*np.sqrt(p[0]*(1-p[0])*p[1]*(1-p[1]))  # ud
        p_matrix[2] = (1-p[0])*p[1] - rho_matrix[0,1]*np.sqrt(p[0]*(1-p[0])*p[1]*(1-p[1]))  # du
        p_matrix[3] = (1-p[0])*(1-p[1]) + rho_matrix[0,1]*np.sqrt(p[0]*(1-p[0])*p[1]*(1-p[1]))  # dd
        return p_matrix

    def get_transition_probs_3d():
        """Calculate transition probabilities for 3-asset case"""
        probs = np.zeros(8)
        moves = list(product([0, 1], repeat=3))
        
        for i, move in enumerate(moves):
            prob = 1.0
            # Base probabilities
            for j, m in enumerate(move):
                prob *= p[j] if m else (1-p[j])
            
            # Correlation adjustments
            for j in range(3):
                for k in range(j+1, 3):
                    if move[j] == move[k]:
                        prob *= (1 + rho_matrix[j,k])
                    else:
                        prob *= (1 - rho_matrix[j,k])
            prob /= 2**3
            probs[i] = prob
        
        # Normalize probabilities
        probs /= np.sum(probs)
        return probs

    # Initialize option values array
    if num_assets == 2:
        values = np.zeros((num_steps+1, num_steps+1))
    else:  # 3 assets
        values = np.zeros((num_steps+1, num_steps+1, num_steps+1))
    
    def get_terminal_payoff(indices):
        """Calculate terminal payoff for given indices"""
        prices = np.array([
            S0_vector[i] * (u[i]**idx) * (d[i]**(num_steps-idx))
            for i, idx in enumerate(indices)
        ])
        return max(0, max(prices) - K)
    
    # Fill terminal values
    if num_assets == 2:
        for i, j in product(range(num_steps+1), repeat=2):
            values[i,j] = get_terminal_payoff([i,j])
        transition_probs = get_transition_probs_2d()
    else:  # 3 assets
        for i, j, k in product(range(num_steps+1), repeat=3):
            values[i,j,k] = get_terminal_payoff([i,j,k])
        transition_probs = get_transition_probs_3d()
    
    # Backward induction
    for step in range(num_steps-1, -1, -1):
        time = step * dt
        is_exercise_date = abs(time - round(time/(T/M))*(T/M)) < dt/2
        
        if num_assets == 2:
            for i, j in product(range(step+1), repeat=2):
                # Current stock prices
                prices = np.array([
                    S0_vector[0] * (u[0]**i) * (d[0]**(step-i)),
                    S0_vector[1] * (u[1]**j) * (d[1]**(step-j))
                ])
                
                # Continuation value
                cont_value = 0
                for k, prob in enumerate(transition_probs):
                    next_i = i + (1 if k < 2 else 0)
                    next_j = j + (1 if k % 2 == 0 else 0)
                    cont_value += prob * values[next_i,next_j]
                cont_value *= np.exp(-r*dt)
                
                # Exercise value
                exercise_value = max(0, max(prices) - K) if is_exercise_date else 0
                values[i,j] = max(exercise_value, cont_value)
                
        else:  # 3 assets
            for i, j, k in product(range(step+1), repeat=3):
                # Current stock prices
                prices = np.array([
                    S0_vector[0] * (u[0]**i) * (d[0]**(step-i)),
                    S0_vector[1] * (u[1]**j) * (d[1]**(step-j)),
                    S0_vector[2] * (u[2]**k) * (d[2]**(step-k))
                ])
                
                # Continuation value
                cont_value = 0
                for m, prob in enumerate(transition_probs):
                    moves = [int(x) for x in format(m, '03b')]
                    next_i = min(i + moves[0], step+1)
                    next_j = min(j + moves[1], step+1)
                    next_k = min(k + moves[2], step+1)
                    cont_value += prob * values[next_i,next_j,next_k]
                cont_value *= np.exp(-r*dt)
                
                # Exercise value
                exercise_value = max(0, max(prices) - K) if is_exercise_date else 0
                values[i,j,k] = max(exercise_value, cont_value)
    
    return values[0,0] if num_assets == 2 else values[0,0,0]


def compute_deltas(S0_vector, h=1e-4, **kwargs):
    """
    Compute deltas for the Bermudan max call option.
    
    Parameters:
    - S0_vector: Initial stock prices for the assets.
    - h: Step size for numerical differentiation.
    - kwargs: Additional arguments for the pricing function.
    
    Returns:
    - deltas: List of deltas for each asset.
    """
    num_assets = len(S0_vector)
    deltas = []
    
    for i in range(num_assets):
        S0_up = S0_vector.copy()
        S0_down = S0_vector.copy()
        
        # Increment and decrement the price of the i-th asset
        S0_up[i] += h
        S0_down[i] -= h
        
        # Calculate option prices
        price_up = price_multi_asset_bermudan_max_call(S0_up, **kwargs)
        price_down = price_multi_asset_bermudan_max_call(S0_down, **kwargs)
        
        # Compute delta
        delta = (price_up - price_down) / (2 * h)
        deltas.append(delta)
    
    return deltas



