#1. Introduction
Simple genetic algorithm in Python.
This class was written very quickly (hence the name) in a couple of hours when I needed to optimise a simple set of parameters to a function (and was a good way to learn some Python back when I was picking it up).
It might serve as the basis for a more comprehensive library in the future, but for now, it's quite basic.

Uses fixed rules for mutation, crossover and selection:

    - 2-point crossover.
    - Gaussian mutation.
    - Random or best selection for reproduction.
    - Worst individuals are dropped each generation.

#2. Usage
Implement an instance of the `GA` class and override the `fitnessfunc` function.
The function should return a fitness value which signifies the quality of a given solution (individual).
Since the GA tries to minimise the function, the smaller the fitness, the better.
The fitness function is, effectively, the function or process you want to optimise.

Use the `GA` constructor to set optimisation parameters.

Call `optimise` and let it work.
You can resume a stopped optimisation by saving the population and then reinitialising the class instance and using the `init_pop` parameter to pass in the saved population.

#3. Dependencies
1. [numpy][1]


[1]: http://www.numpy.org/
