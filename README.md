# SemiStaticReplicationNeuralNet

## Project Sunmmary

In this project, we replicate and extend the framework by Lokeswarh et al. for pricing and hedging path-dependent options that combines regress-later Monte Carlo with shallow neural networks (RLNN), enabling both semi-static and fully static replication strategies. By training simple feedforward networks on simulated paths, we obtain closed-form approximations for Bermudan-style continuation values and explicit decomposition into simple European options.

In the single-asset case, each hidden neuron corresponds to a short-maturity call, put, or forward payoff, allowing transparent, interpretability-driven hedging. Extensive numerical experiments demonstrate that the RLNN approach produces highly accurate price estimates—comparable to or better than classical binomial and Monte Carlo benchmarks in both pricing and hedging—while drastically reducing rebalancing frequency.

In multi-asset settings, the RLNN framework remains robust, with arithmetic basket and max option payoffs managed via a combination of correlated path simulation and log stock price inputs, enabling a continuation value approximation based on Gaussian properties. Finally, in selected stocks, we show how to adapt RLNN-implied theoretically optimal portfolios into tradable portfolios of liquid market instruments, either via direct strike-mapping or by constraining the neural network architecture to tradable instruments and refitting such that the replicating portfolio corresponding to the constrained neural network solution directly matches a tradable portfolio for hedging simple European options.

Although this approach does not outperform a simple min-max linear programming benchmark using implied strikes around the target, the implied instrument combination from the neural network fit can be used to inform the selection of the initial subset of instruments for the linear programming problem, reducing the complexity of the linear programming solution and the opaqueness of choosing an initial subset manually in the case of hedging a complex derivative.

For future work, we recommend expanding our proposed tradable-constrained neural network structure aproach into a comprehensive hedging experiment for more complex derivatives. Overall, the RLNN framework provides a scalable and interpretable solution for managing complex derivative portfolios, delivering reduced hedging variance, lower transaction costs, and precise price bounds for both single- and multi-asset derivatives.
