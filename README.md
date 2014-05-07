Simple genetic algorithm in Python.

Uses fixed rules for mutation, crossover and selection:

    - 2-point crossover.
    - Gaussian mutation.
    - Random or best selection for reproduction.
    - Worst individuals are dropped each generation.

# Usage
Implement an instance of the `GA` class and override the `fitnessfunc` function.

Use the `GA` constructor to set optimisation parameters.
