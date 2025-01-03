import matplotlib.pyplot as plt 
from parallel_RLNN import *
import numpy as np


def portfolio_value(weights, stock_price, r, sigma, dt, S0):
    
    weights_layer1 = weights[0].flatten()
    weights_layer2 = weights[2].flatten()
    bias_layer1 = weights[1]
    bias_layer2 = weights[3][0]
    St = stock_price/S0
    
    cost = bias_layer2
    for i in range(len(weights_layer1)):
        wi, bi = weights_layer1[i], bias_layer1[i]
        
        if wi >= 0 and bi >= 0:
            # Forward Price
            temp_cost = wi * St * np.exp(r * dt) + bi
        elif wi > 0 and bi < 0:
            # European Call Option
            strike = -bi / wi
            temp_cost = wi * black_scholes(St, strike, r, sigma, dt, option_type='call')
        elif wi < 0 and bi > 0:
            # European Put Option
            strike = -bi / wi
            temp_cost = -wi * black_scholes(St, strike, r, sigma, dt, option_type='put')
        else:
            # Expected value is 0
            temp_cost = 0.0
            
        cost += temp_cost * weights_layer2[i]
    
    return cost * np.exp(-r * dt)

def check_payoffs(stock_price, weights, S0):
    """
    Given a single stock_price, compute the total 
    payoff implied by the 2-layer NN with piecewise logic.
    
    :param stock_price:  current price of the underlying (float)
    :param weights:      a tuple/list containing 
                         [weights_layer1, bias_layer1, weights_layer2, bias_layer2]
    :param S0:           normalizing constant (e.g. initial stock price)
    :return:             total payoff (float)
    """
    payoffs = 0.0    
    # Extract layers
    weights_layer1 = weights[0].flatten()  # array of w1_i
    bias_layer1    = weights[1]  # array of b1_i
    weights_layer2 = weights[2].flatten()  # array of w2_i
    bias_layer2    = weights[3][0]  # float or array of length 1

    # Normalize
    normalized_stock = stock_price / S0
    
    # Start from second-layer bias
    # (if bias_layer2 is an array, do bias_layer2[0])
    payoffs = bias_layer2

    # Loop over all first-layer neurons
    for i in range(len(weights_layer1)):
        w1i = weights_layer1[i]
        b1i = bias_layer1[i]
        w2i = weights_layer2[i]

        # Case 1: w1 >= 0, b1 >= 0 => Forward contract
        if w1i >= 0 and b1i >= 0:
            payoffs += w2i * (w1i * normalized_stock + b1i)

        # Case 2: w1 > 0, b1 < 0 => call-like payoff
        elif w1i > 0 and b1i < 0:
            strike_call = -b1i / w1i
            call_payoff = max(normalized_stock - strike_call, 0)
            payoffs += w2i * w1i * call_payoff

        # Case 3: w1 < 0, b1 > 0 => put-like payoff
        elif w1i < 0 and b1i > 0:
            strike_put = -b1i / w1i
            put_payoff = max(strike_put - normalized_stock, 0)
            payoffs -= w2i * w1i * put_payoff

        else:
            # The "else" might be w1 <= 0, b1 <= 0 => payoff is effectively 0
            0.0

    return payoffs

def cal_continuation_value2(weights, no_hidden_units, stock_prices, r, sigma, dt, normalizer):
    """ Calculate the continuation value of the Bermudan option

    Args:
        w1 (array): Weights of the first layer
        b1 (array): Biases of the first layer
        w2 (array): Weights of the second layer
        b2 (float): Biases of the second layer
        no_hidden_units (int): Number of hidden units
        stock_prices (float): Stock prices at time
        r (float): Risk free rate
        sigma (float): volatility
        dt (_type_): time step
        M (_type_): number of simulations(samples)
        normalizer (_type_): normalizer for the stock prices

    Returns:
        _type_: _description_
    """
     ## Get the weights of the model
    w1 = weights[0].reshape(-1)
    b1 = weights[1].reshape(-1)
    w2 = weights[2].reshape(-1)
    b2 = weights[3].reshape(-1)
    normalized_stock_values = stock_prices/normalizer
    continuation_value = 0
    for i in range(no_hidden_units):
        continuation_value += expected_value(w1[i], b1[i], normalized_stock_values, r, sigma, dt) * w2[i]
        #print(continuation_value)
    continuation_value += b2
    
    return continuation_value * np.exp(-r*dt)

def semi_static_hedging_simulation(stock_paths, weights, r, sigma, T, M, S0, K, price, hidden_nodes):
    N = len(stock_paths)
    dt = T / M
    hedging_errors = np.zeros(N, dtype=float)

    for i in range(N):
        cash_account = price
        weights_now = weights[-1]
        
        # Initial portfolio setup
        cash_account -= portfolio_value(weights_now, stock_paths[i][0], r, sigma, dt, S0)
        
        #print("cash account at the start", cash_account)
        
        for j in range(1, M):
            cash_account *= np.exp(r * dt)
            
            Exercise_value = payoff_fun(stock_paths[i][j], K, "put")
            weights_next = weights[-(j+1)]
            cont_value = cal_continuation_value2(weights_next, hidden_nodes, stock_paths[i][j], r, sigma, dt, S0)
            
            if Exercise_value > cont_value:
                # At exercise: get payoff from current portfolio and exercise
                portfolio_t = check_payoffs(stock_paths[i][j], weights_now, S0)
                cash_account += portfolio_t
                hedging_errors[i] = cash_account - Exercise_value
                break
            
            # Rebalancing: get payoff from old portfolio, pay for new one
            portfolio_t = check_payoffs(stock_paths[i][j], weights_now, S0)
            cash_account += portfolio_t
            
            weights_now = weights[-(j+1)]
            cash_account -= portfolio_value(weights_now, stock_paths[i][j], r, sigma, dt, S0)
        else:
            cash_account *= np.exp(r * dt)
            portfolio_t = check_payoffs(stock_paths[i][M], weights_now, S0)
            cash_account += portfolio_t
            payoff_T = payoff_fun(stock_paths[i][M], K, "put")
            hedging_errors[i] = cash_account - payoff_T
            
    return hedging_errors



