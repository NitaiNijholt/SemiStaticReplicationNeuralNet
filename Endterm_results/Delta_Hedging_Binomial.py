import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time 
from tqdm import tqdm
import pandas as pd


def stock_price_simulator(S0, mu, sigma, T, N, M):
    """_summary_

    Args:
        S0 (_type_): initial stock price
        mu (_type_): risk free rate
        sigma (_type_): volatility
        T (_type_): time to maturity
        N (_type_): number of time steps
        M (_type_): number of simulations

    Returns:
        S : Stock price Paths of M simulations
    """
    dt = T/N
    S = np.zeros((M, N+1))
    S[:, 0] = S0
    for i in range(1, N+1):
        S[:, i] = S[:, i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(loc=0.0, scale=1.0, size=M))
    
    return S

def payoff_fun(S, K, option_type):
    """

    Args:
        S (_type_): Stock price
        K (_type_): Strike price
        option_type (_type_): type of option, call or put
    Returns:
        payoff : payoff of the option
    """
    if option_type == 'call':
        return np.maximum(S - K, 0)
    elif option_type == 'put':
        return np.maximum(K - S, 0)
    
    
def black_scholes(S, K, r, sigma, T, option_type='call'):
    """ 
    Calculate the price of a European option using Black-Scholes formula

    Args:
        S (_type_): Initial Stock price
        K (_type_): Strike price
        r (_type_): risk free rate
        sigma (_type_): volatility
        T (_type_): time to maturity
        option_type (str, optional): type of option .

    Returns:
        price of the option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), norm.cdf(d1)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), -norm.cdf(-d1)
    

def bermudan_option_binomial(S0, K, T, r, sigma, N, exercise_dates, option_type="put"):
    """
    Price a Bermudan option and calculate delta using the Binomial Tree method.
    
    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - N: Number of steps in the binomial tree
    - exercise_dates: List of times (as fractions of T) at which early exercise is allowed
    - option_type: "call" for call option, "put" for put option
    
    Returns:
    - Tuple of (option_price, delta)
    """
    
    # Calculate parameters for binomial tree
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    stock_prices = np.zeros(N + 1)
    stock_prices[0] = S0 * d**N
    for i in range(1, N + 1):
        stock_prices[i] = stock_prices[i - 1] * u / d

    # Initialize option values at maturity
    option_values = np.maximum(0, (stock_prices - K) if option_type == "call" else (K - stock_prices))
    
    # Backward induction
    for i in range(N - 1, -1, -1):
        # Step back in time for stock prices
        stock_prices = stock_prices[:-1] * u
        
        # Calculate continuation values
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])
        
        # Apply early exercise at specified dates
        if i * dt in exercise_dates:
            exercise_values = np.maximum(0, (stock_prices - K) if option_type == "call" else (K - stock_prices))
            option_values = np.maximum(option_values, exercise_values)
        
        # Calculate delta at the first step
        if i == 1:
            # Store option values for delta calculation
            option_up = option_values[1]
            option_down = option_values[0]
            stock_up = S0 * u
            stock_down = S0 * d
            
    # Calculate delta using finite difference
    delta = (option_up - option_down) / (stock_up - stock_down)
    
    return option_values[0], delta


def calculate_hedge_error(S0, K, mu, sigma, T, hedge_freq, monitoring_dates, hedging_dates, M = 10000):
    hedge_error = np.zeros(M)
    for m in tqdm(range(M)):
        V0, delta = bermudan_option_binomial(S0, K, T, mu, sigma, 200, monitoring_dates, option_type="put")
        #print(f'Price of the option: {V0}', f'Delta: {delta}')
        cash_flow = V0 - delta * S0
        No_of_shares = delta
        updated_monitoring_dates = monitoring_dates
        #print(f'Initial cash flow: {cash_flow}', f'Number of shares to buy: {No_of_shares}')
        S = stock_price_simulator(S0, mu, sigma, T, hedge_freq, 1).flatten()
        #print(S)
        dt = T/hedge_freq
        #print("time stpes in hedging dates", dt)
        for i in range(1, len(hedging_dates)- 1):
            #print(f'Hedging date: {i*dt}')
            updated_monitoring_dates = updated_monitoring_dates - dt
            updated_monitoring_dates = updated_monitoring_dates[updated_monitoring_dates > 0]    
            #print(f'Updated monitoring dates: {updated_monitoring_dates}')
            V1, delta2 = bermudan_option_binomial(S[i], K, updated_monitoring_dates[-1], mu, sigma, 100, updated_monitoring_dates, option_type="put")
            #V1, delta2 = binomial_option_pricing(S[i], K, updated_monitoring_dates[-1], mu, sigma, 100, updated_monitoring_dates, option_type='put')
            if i*dt in monitoring_dates:
                #print("inside", i*dt)
                #print(f'Price of stock: {S[i]}', f'Price of the option: {V1}', f'K: {K}', f'payoff : {payoff_fun(S[i], K, "put")}')
                if payoff_fun(S[i], K, 'put') > V1:
                    #print("Exercise")
                    break
                cash_flow = cash_flow * np.exp(mu*dt) - (delta2 - No_of_shares) * S[i]
                No_of_shares = delta2
                #print(f'Cash flow: {cash_flow}', f'Number of shares to buy: {No_of_shares}')
            else:
                cash_flow = cash_flow * np.exp(mu*dt) - (delta2 - No_of_shares) * S[i]
                No_of_shares = delta2
                #print("Not in monitoring date")
            #print("i", i, No_of_shares)
        if i == len(hedging_dates)-2:
            #print("Last date", i)
            payoff_last = payoff_fun(S[i+1], K, 'put')   
            final_cash_flow = cash_flow * np.exp(mu*dt) + No_of_shares * S[i+1]
        else:
            #print("i at end", i*dt, V1, payoff_fun(S[i], K, 'put'))
            payoff_last = payoff_fun(S[i], K, 'put')
            final_cash_flow = cash_flow * np.exp(mu*dt) + No_of_shares * S[i]
        #print(f'Final Hedge Error: {payoff_last - final_cash_flow}')
        hedge_error[m] = payoff_last - final_cash_flow
        
    return hedge_error

