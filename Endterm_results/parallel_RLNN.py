import numpy as np
from scipy.stats import norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split



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
        S[:, i] = S[:, i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal(size=M))
    
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
    
def create_shallow_NN(input_dim, hidden_units):
    """
    Create a shallow neural network with 1 hidden layer

    Args:
        input_dim (int): number of nodes in input layer
        hidden_units (int): number of nodes in hidden layer

    Returns:
        model : Neural network model
    """
    model = Sequential()
    # Use Input layer for specifying input shape
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units, activation='relu', kernel_initializer='random_uniform', bias_initializer= 'random_uniform'))
    model.add(Dense(1, activation='linear', kernel_initializer='random_normal', bias_initializer= 'random_uniform'))
    
    return model

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
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
def expected_value(wi, bi, Stm_1, r, sigma, dt):
    """
    Calculate E[max(wi * Stm + bi, 0) | St] based on the cases provided.

    Parameters:
    - wi : float, weight parameter
    - bi : float, bias parameter
    - Stm : float, price at time t_m

    Returns:
    - float, expected value based on the case
    """
    if wi >= 0 and bi >= 0:
        # Case 1: Price of a forward contract
        ### Forward Price Impelmentation
        return wi * Stm_1 * np.exp(r * dt) + bi

    elif wi > 0 and bi < 0:
        # Case 2: Value of a European call option
        strike = -bi / wi
        # Using max(0, Stm - strike) to approximate the expectation
        ### Implement
        return wi * black_scholes(Stm_1, strike, r, sigma, dt, option_type='call')

    elif wi < 0 and bi > 0:
        # Case 3: Value of a European put option
        strike = -bi / wi
        return - wi * black_scholes(Stm_1, strike, r, sigma, dt, option_type='put')

    elif wi <= 0 and bi <= 0:
        # Case 4: Expected value is 0
        return 0.0
    
def cal_continuation_value(w1, b1, w2, b2, no_hidden_units, stock_prices, r, sigma, dt, M, normalizer):
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
    normalized_stock_values = stock_prices/normalizer
    continuation_value = np.zeros(M)
    for j in range(M):
        for i in range(no_hidden_units):
            continuation_value[j] += expected_value(w1[i], b1[i], normalized_stock_values[j], r, sigma, dt) * w2[i]
        continuation_value[j] += b2
    
    return continuation_value * np.exp(-r*dt)




def RLNN_Algo(S0, K, r, vol, sample_size, no_mon, T, style, no_hidden_units, l_rate2):
    """
    Args:
        S0 (): Initial stock price
        K (_type_): Stirke price
        r (_type_): risk free rate
        vol (_type_): volatility
        sample_size (_type_): number of samples to generate
        no_mon (_type_): Number of monitoring dates
        style (_type_): option type
        no_hidden_units (_type_): number of hidden units in the neural network
    """
    
    weights = []
    errors = []
    N = no_mon
    
    stock_paths = stock_price_simulator(S0, r, vol, T, no_mon, sample_size)
    ## Initialize the option price matrix
    option_price = np.zeros(stock_paths.shape)
    ## Calculate the option price at the maturity date
    option_price[:, N] = payoff_fun(stock_paths[:, N], K, style)
    payoff_T = payoff_fun(stock_paths[:, N], K, style)
     
    X = stock_paths[:, N].reshape(-1, 1)
    ## Normalize the stock prices
    normalizer = S0
    normalized_stock_paths = stock_paths[:, N]/normalizer
    ## X_train and y_train for the neural network
    X = normalized_stock_paths.reshape(-1, 1)
    y = payoff_T.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Compile the model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    model2  = create_shallow_NN(1, no_hidden_units)
    model2.compile(loss='mean_squared_error', optimizer=Adam(l_rate2))
    ## Train the model  
    model2.fit(X_train, y_train, epochs=3000, batch_size=int(0.1 * X_train.shape[0]), 
              validation_data=(X_test, y_test), callbacks=[es], verbose = 0)
    
    # Test the model
    y_hat_val = model2.predict(X_test)
    
    errors.append(np.mean((y_test - y_hat_val)**2)) ## Store the errors for Last monitoring date
    
    ## Get the weights of the model
    w1 = model2.layers[0].get_weights()[0].reshape(-1)
    b1 = model2.layers[0].get_weights()[1].reshape(-1)
    w2 = model2.layers[1].get_weights()[0].reshape(-1)
    b2 = model2.layers[1].get_weights()[1].reshape(-1)
    
    ## Store the weights 
    weights.append(model2.get_weights())
    
    ## Calculate the continuation value 
    continuation_value = cal_continuation_value(w1, b1, w2, b2, no_hidden_units, stock_paths[:, N-1], r, vol, T/N, sample_size, normalizer)
    ##print(continuation_value[continuation_value < 0])
    fun_h = payoff_fun(stock_paths[:, N-1], K, style)               # Payoff at the previous monitoring date
    if N == 1:
        option_price[:, N-1] = continuation_value
    else:
        option_price[:, N-1] = np.maximum(fun_h, continuation_value)    # Option price at the previous monitoring date
    
    
    ## Calculate the option price at the previous monitoring dates
    
    model2.optimizer.learning_rate.assign(0.01)
    
    for i in range(N-1, 0, -1):
        normalizer = S0
        normalized_stock_paths = stock_paths[:, i]/normalizer
        X = normalized_stock_paths.reshape(-1, 1)
        y = option_price[:, i].reshape(-1, 1)
        es2 = EarlyStopping(monitor='val_loss', mode='min', patience=20,restore_best_weights=True, min_delta=0.00001, verbose=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model2.fit(X_train, y_train, epochs=3000, batch_size=int(0.1 * X_train.shape[0]), 
                  validation_data=(X_test, y_test), callbacks=[es2], verbose =0)
        
        y_hat = model2.predict(X_test)
        
        errors.append(np.mean((y_test - y_hat)**2))    ## Store the errors for the previous monitoring dates
        
        # compute the continuation value
        w1 = model2.layers[0].get_weights()[0].reshape(-1)
        b1 = model2.layers[0].get_weights()[1].reshape(-1)
        w2 = model2.layers[1].get_weights()[0].reshape(-1)
        b2 = model2.layers[1].get_weights()[1].reshape(-1)
        weights.append(model2.get_weights())
        continuation_value = cal_continuation_value(w1, b1, w2, b2, no_hidden_units, stock_paths[:, i-1], r, vol, T/N, sample_size, normalizer)
        fun_h = payoff_fun(stock_paths[:, i-1], K, style)            # Payoff at the previous monitoring date
        if i == 1:
            option_price[:, i-1] = continuation_value
        
        else:
            option_price[:, i-1] = np.maximum(fun_h, continuation_value)    
        
    return option_price[0,0] , weights, errors

#
def calculate_lower_bound(stock_paths, K, r, sigma, T, no_mon_dates, option_type, trained_weights, nodes):
    """_summary_

    Args:
        S0 (_type_): _description_
        K (_type_): _description_
        r (_type_): _description_
        sigma (_type_): _description_
        T (_type_): _description_
        no_mon_dates (_type_): Number of monitoring dates
        option_type (_type_): _description_

    Returns:
        _type_: _description_
    """
    S0 = stock_paths[0, 0]
    dt = T / no_mon_dates
    M = stock_paths.shape[0]
    
    tau = np.full(stock_paths.shape[0], no_mon_dates-1)
    h_S = payoff_fun(stock_paths[:, no_mon_dates-1], K, option_type)
    
    for m in range(no_mon_dates-1):
        normalizer = S0
        St = stock_paths[:, m]    
        payoff_t = payoff_fun(St, K, option_type)
        current_weights = trained_weights[no_mon_dates - m - 2]
        weights_layer_1 = np.array(current_weights[0]).reshape(-1)
        biases_layer_1 = np.array(current_weights[1])
        weights_layer_2 = np.array(current_weights[2]).reshape(-1)
        biases_layer_2 = np.array(current_weights[3])
        continuation_value = cal_continuation_value(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, nodes, St, r, sigma, dt, M, normalizer)
        exceed = np.logical_and(payoff_t > continuation_value, tau > m)
        tau[exceed] = m
        h_S[exceed] = payoff_t[exceed]
        
        monitoring = np.arange(0, 1.01, 1/no_mon_dates)
        discounted_values = np.zeros(M, dtype=float)
        for j in range(M):
            discounted_values[j] = h_S[j] * np.exp(-r  * monitoring[tau[j]])

        lower_bound = np.mean(discounted_values)

    return lower_bound     

def upper_bound(rfr, vol, trained_weights, stock_paths, strike, monitoring, style, nodes):
    
    sample_size = len(stock_paths[:, 0])
    n_mon = len(monitoring)
    differences = np.diff(monitoring)

    b = np.exp(- rfr * np.cumsum(differences))
    b = np.insert(b, 0, 1)
    S0 = stock_paths[0, 0]

    rlnn = create_shallow_NN(1, nodes)

    martingale = np.zeros((sample_size, n_mon))

    for m in range(1, n_mon):
        normalizer = S0
        current_weights = trained_weights[- m]
        rlnn.set_weights(current_weights)
        w1 = np.array(current_weights[0]).reshape(-1)
        b1 = np.array(current_weights[1])
        w2 = np.array(current_weights[2]).reshape(-1)
        b2 = np.array(current_weights[3])
        q = cal_continuation_value(w1, b1, w2, b2, nodes, stock_paths[:, m - 1], rfr, vol, differences[m - 1], sample_size, normalizer)

        q_part = q * b[m - 1]
        g_part = (rlnn.predict(stock_paths[:, m] / normalizer, verbose=0) * b[m]).reshape(-1)

        martingale[:, m] = (g_part - q_part)

    martingale = np.cumsum(martingale, axis=1)

    payoffs = payoff_fun(stock_paths, strike, style)
    upr = np.mean(np.max(payoffs * b - martingale, axis=1))

    return upr


