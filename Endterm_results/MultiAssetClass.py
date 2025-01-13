import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import tensorflow as tf


def generate_covarinace_mat(cor_mat, volatilities, dt):
    """
    Generates the covariance matrix for the correlated stock price simulation.

    Parameters:
        cor_mat (np.ndarray): Correlation matrix of the stock returns.
        volatilities (np.ndarray): Array of volatilities for the stocks.
        dt (float): Time increment.

    Returns:
        np.ndarray: Covariance matrix for the stock returns.
    """
    vol_diag = np.diag(volatilities)
    return dt * vol_diag @ cor_mat @ vol_diag


def generate_multi_stock_price(S0, r, volatilities, cor_mat, dt, N, q):
    """
    Simulates the evolution of multiple stock prices using a correlated geometric Brownian motion model.

    Parameters:
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        volatilities (np.ndarray): Array of volatilities for each stock.
        cor_mat (np.ndarray): Correlation matrix of the stock returns.
        dt (float): Time increment.
        N (int): Number of time steps to simulate.
        q (float): Continuous dividend yield.

    Returns:
        np.ndarray: Simulated stock prices for all time steps and stocks (shape: [number_of_stocks, N]).
    """
    # Initialize variables
    q_vect = np.ones(len(volatilities)) * q
    stock_prices = np.zeros((len(volatilities), N))
    stock_prices[:, 0] = np.log(S0) * np.ones(len(volatilities))
    mu = stock_prices[:, 0] + (r - q_vect - 0.5 * volatilities ** 2) * dt

    # Generate the covariance matrix
    Sigma = generate_covarinace_mat(cor_mat, volatilities, dt)

    # Simulate stock prices over time
    for i in range(1, N):
        stock_prices[:, i] = np.random.multivariate_normal(mu, Sigma)
        current_log_prices = stock_prices[:, i]
        mu = current_log_prices + (r - q - 0.5 * volatilities ** 2) * dt

    # Return prices in their exponential form (convert log prices to actual prices)
    return np.exp(stock_prices)

def arithmatic_basket_option_price(S0, r, volatilities, cor_mat, T, N, weights, K, M):
    """
    Calculates the price of an arithmetic basket option using Monte Carlo simulation.

    Parameters:
        S0 (float): Initial stock price for all stocks (assumed the same for simplicity).
        r (float): Risk-free interest rate.
        volatilities (np.ndarray): Array of volatilities for the stocks.
        cor_mat (np.ndarray): Correlation matrix of the stock returns.
        dt (float): Time increment.
        N (int): Number of time steps to simulate.
        weights (np.ndarray): Array of weights for the basket components.
        K (float): Strike price of the option.
        M (int): Number of Monte Carlo simulations.

    Returns:
        float: Estimated price of the arithmetic basket option.
    """
    option_prices = []
    q_vect = np.zeros(len(volatilities))
    for _ in range(M):
        stock_prices = generate_multi_stock_price(S0, r, volatilities, cor_mat, T, N, q_vect)
        price_T = stock_prices[:, 1]
        avg_ST = np.dot(weights.T, price_T)
        option_price = np.maximum(K - avg_ST, 0)
        option_prices.append(option_price)
    option_price = np.mean(option_prices) * np.exp(-r * T)
    
    return option_price


def price_max_call_option(S0, r, volatilities, cor_mat, dt, N, K, M):
    option_prices = []
    for _ in range(M):
        stock_prices = generate_multi_stock_price(S0, r, volatilities, cor_mat, dt, N)
        price_T = stock_prices[:, 1]
        option_price = np.maximum(np.max(price_T) - K, 0)
        option_prices.append(option_price)
    option_price = np.mean(option_prices) * np.exp(-r * dt)
    
    return option_price

def multi_asset_NN(no_hidden_nodes):
    
    model = Sequential()
    model.add(Dense(no_hidden_nodes, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

def calculate_continuation_value_multi(weights1, biases1, weights2, bias2, stock_prices, 
                                    volatilities, covariance_matrix, risk_free_rate, 
                                    time_step, dividend_yield):
    """
    Calculate continuation values for multiple assets using a neural network approach.
    
    Parameters:
    -----------
    weights1: ndarray
        First layer weights matrix
    biases1: ndarray
        First layer bias vector
    weights2: ndarray
        Second layer weights vector
    bias2: float
        Second layer bias scalar
    stock_prices: ndarray
        Matrix of stock prices (samples × assets)
    volatilities: ndarray
        Vector of volatilities for each asset
    covariance_matrix: ndarray
        Covariance matrix of asset returns
    risk_free_rate: float
        Risk-free interest rate
    time_step: float
        Time step size (dt)
    dividend_yield: float
        Dividend yield
    
    Returns:
    --------
    ndarray
        Continuation values for each sample
    """
    # Get dimensions
    num_samples = stock_prices.shape[0]
    num_hidden_nodes = weights1.shape[1]
    num_assets = stock_prices.shape[1]
    
    # Create dividend yield vector for all assets
    dividend_yields = np.tile(dividend_yield, (1, num_assets))
    
    # Calculate log-adjusted stock prices
    drift_adjustment = (risk_free_rate - dividend_yields - 0.5 * np.square(volatilities)) * time_step
    drift_matrix = np.tile(drift_adjustment.reshape(1, num_assets), (num_samples, 1))
    log_adjusted_prices = np.log(stock_prices) + drift_matrix
    
    # Initialize option values
    option_values = np.zeros((num_samples, 1))
    
    # Calculate for each hidden node
    for node in range(num_hidden_nodes):
        # Get weights for current node and reshape
        node_weights = weights1[:, node].reshape(num_assets, 1)
        
        # Calculate mean (mu)
        mu = np.dot(log_adjusted_prices, node_weights) + biases1[node]
        
        # Calculate standard deviation
        variance = np.dot(np.dot(node_weights.T, covariance_matrix), node_weights)
        std_dev = np.sqrt(variance)
        
        # Calculate first term (ft)
        first_term = mu * norm(0, std_dev).cdf(mu)
        
        # Calculate second term (st)
        second_term = (std_dev / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (mu / std_dev) ** 2)
        
        # Update option values
        option_values += weights2[node] * (first_term + second_term)
    
    # Calculate final continuation value with discounting
    discount_factor = np.exp(-(risk_free_rate - dividend_yield) * time_step)
    continuation_value = (option_values + bias2) * discount_factor
    
    return continuation_value

def payoff_func_Multi(stock_prices, w, K, type='Basket'):
    if type == 'Basket':
        payoff = np.maximum(K - np.dot(stock_prices, w), 0)
    elif type == 'Max':
        payoff = np.maximum(np.max(stock_prices, axis=1).flatten() - K, 0)
    return payoff

def RLNN_pre_training(model, S0, r, vol_list, cor_mat, dt, N, M, K, no_of_assets,w, q, type='Basket'):
    
    stock_prices = np.zeros((M, no_of_assets, N))
    for i in range(M):
        stock_prices[i] = generate_multi_stock_price(S0, r, vol_list, cor_mat, dt, N, q)
        
    stock_price_T = stock_prices[:, :, -1]
    X_train = np.log(stock_price_T)
    y_train = payoff_func_Multi(stock_price_T, w, K, type)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    model.fit(X_train, y_train, epochs=3000, validation_split=0.2, callbacks=[es], verbose=0)
    
    mse = model.evaluate(X_train, y_train)
    
    return model, mse

def RLNN_MultiAsset(M, no_of_assets, K, r, dt, divident, model, cor_mat, vol_list, weights, no_of_exercise_days, S0, l_rate2, type_option='Max'):
   """M: Number of paths
      N: Number of time steps
      no_of_assets: Number of assets
      K: Strike price
      model: Neural network model"""
   # Creating zero n-d arrays for intrinsic value, continuation value and option value
   mse = np.zeros(no_of_exercise_days)
   N = no_of_exercise_days + 1
   model, _ = RLNN_pre_training(model, S0, r, vol_list, cor_mat, dt, N, M, K, no_of_assets,weights, divident, type_option)
   model.optimizer.learning_rate.assign(l_rate2)
   model, _ = RLNN_pre_training(model, S0, r, vol_list, cor_mat, dt, N, M, K, no_of_assets, weights, divident, type_option)
   cov_mat = generate_covarinace_mat(cor_mat, vol_list, dt)
   ## Main Training Starts Here
   #model.optimizer.learning_rate.assign(1e-3)
   stock_prices = np.zeros((M, no_of_assets, N))
   for i in range(M):
      stock_prices[i] = generate_multi_stock_price(S0, r, vol_list, cor_mat, dt, N, divident)
      
   # Creating zero n-d arrays for intrinsic value, continuation value and option value
   no_of_paths = M
   continuation_value = np.zeros((no_of_paths, 1))
   stock_vec = stock_prices[:, :,  no_of_exercise_days]
   intrinsic_value = payoff_func_Multi(stock_vec, weights, K, type_option)
   continuation_value = intrinsic_value
   es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
   nn_weights = []
   for day in range(no_of_exercise_days - 1, -1, -1):
      
      stock_vec = stock_prices[:, :, day + 1]
      option_value = continuation_value
      
      X = np.log(stock_vec)
      Y = option_value
      if type_option == 'Max':
         Y = Y.flatten()
    
      # Split data into train+val and test sets
      #X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
      # Further split train+val into training and validation sets
      X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)  # 0.25 x 0.8 = 0.2

      # Train the model
      if type_option == 'Max':
         Y_train = Y_train.flatten()
         Y_val = Y_val.flatten()
      #print("shape of X_train", X_train.shape, "shape of Y_train", Y_train.shape)
      #tf.keras.utils.set_random_seed(2)
      model.fit(X_train, Y_train, epochs=3000, validation_data=(X_val, Y_val), callbacks=[es], verbose=0, batch_size=int(M/10))
      #Evaluate the model
      
      mse[int(day)] = model.evaluate(X_train, Y_train)
      print(f"mse at {day}", model.evaluate(X_train, Y_train))
      
      w1 = np.array(model.layers[0].get_weights()[0])
      w2 = np.array(model.layers[1].get_weights()[0])
      bias1 = np.array(model.layers[0].get_weights()[1])
      bias2 = np.array(model.layers[1].get_weights()[1])
      stock_vec = stock_prices[:, :,  day]
      
      nn_weights.append([w1, bias1, w2, bias2])
      
      continuation_value = calculate_continuation_value_multi(w1, bias1, w2, bias2, stock_vec, vol_list, cov_mat, r, dt, divident)
      #print("day", day)
   #print(np.mean(continuation_value))
   
   return np.mean(continuation_value), mse, nn_weights

