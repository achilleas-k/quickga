"""
    Simple genetic algorithm class in Python

TODO:
    - INSTRUCTIONS (note that genemax is INCLUSIVE)
    - Multithread fitness evaluation
    - Errors shouldn't call exit - use exceptions
    - Parameterise everything (max age, individuals to keep on reset)
    - Variable length chromosomes
    - More crossover, mutation and selection methods
    - Parallel fitness evaluations
    - Reinsert best into population when resetting?
    - Save progress and current configuration in a different file (not log)
    that can be loaded directly from the class to continue the search (pickle?)
"""
from sys import exit
import copy
import numpy as np


class GA:

    def __init__(self, max_population, chromlength,
                 mutation_probability=0.01, mutation_strength=0.1,
                 selection_percentage=0.2,
                 genemin=-1.0, genemax=1.0,
                 init_pop=None, genetype=float, logfile="quickga.log"):
        self.max_population = max_population
        self.mutation_probability = mutation_probability
        self.mutation_strength = mutation_strength
        self.selection_percentage = selection_percentage
        self.chromlength = chromlength
        self.genebounds = (genemin, genemax)
        self.population = init_pop
        self.genetype = genetype
        self.logfile = open(logfile, 'w')
        if issubclass(self.genetype, bool):
            exit("NOT IMPLEMENTED: Boolean gene types not implemented yet")
        if self.population is None:
            self.population = []
            self.init_population()

    def init_population(self):
        del(self.population)
        self.population = []
        low, high = self.genebounds
        if issubclass(self.genetype, float):
            randfunc = lambda size: np.random.random(size)*(high-low)+low
        elif issubclass(self.genetype, int):
            randfunc = lambda size: np.random.randint(low, high+1, size)
        elif issubclass(self.genetype, bool):
            exit("NOT IMPLEMENTED: Boolean gene types not implemented yet")
        for p in range(self.max_population):
            randchrom = randfunc(self.chromlength)
            newind = self.Individual(randchrom, 0)
            self.population.append(newind)

    def fitnessfunc(self):
        """
        Calculates the fitness value of an individual, stores it in the
        individual's `fitness` member and returns it.
        This function is the target function to be optimised by the GA.

        This function should be overwritten by any implementation of the
        class.

        TODO: This could be done better. Have the storing and
        returning parts coded in the function and have another function,
        `evaluate` the fitness of an arbitrary chromosome, with no connection
        to the classes in this file.
        """

    def evaluate_population(self, optargs):
        """
        Updates the fitness values of the population, for all individuals
        flagged as unevaluated.
        """
        for ind in self.population:
            if ind.fitness is None:
                self.fitnessfunc(ind, *optargs)

    def sort_population(self, optargs):
        """
        Sorts the population based on fitness values such that the first
        individual is the best in the population.
        This method first calls the `evaluate_population` method to calculate
        the fitness values of any unevaluated individuals
        """
        self.evaluate_population(optargs)
        self.population.sort(key=lambda individual: individual.fitness)

    def crossover(self, parent_one, parent_two):
        """
        Two-point crossover.
        """
        parchrom_one = copy.deepcopy(parent_one.chromosome)
        parchrom_two = copy.deepcopy(parent_two.chromosome)
        if parent_one.chromlength != parent_two.chromlength:
            exit("ERROR: Chromosome lengths don't match. "
                        "Skipping crossover.")

        cutpoint_one = cutpoint_two = 0
        chromlength = parent_one.chromlength
        cutpoints = np.random.choice(range(chromlength), 2, replace=False)
        cutpoint_one = min(cutpoints)
        cutpoint_two = max(cutpoints)

        childchrom_one = np.append(parchrom_one[0:cutpoint_one],
                            parchrom_two[cutpoint_one:cutpoint_two])
        childchrom_one = np.append(childchrom_one, parchrom_one[cutpoint_two:])
        childchrom_two = np.append(parchrom_two[0:cutpoint_one],
                            parchrom_one[cutpoint_one:cutpoint_two])
        childchrom_two = np.append(childchrom_two, parchrom_two[cutpoint_two:])
        if childchrom_one.size != childchrom_two.size:
            exit("ERROR: Child chromosome lengths don't match."
                 " This shouldn't happen")
        if not (childchrom_one.size == parchrom_one.size ==
                childchrom_two.size == parchrom_two.size):
                    exit("ERROR: Chromosome lengths changed during crossover")

        # Turn childchroms into individuals
        # Childrens' generation will be youngest parent's gen+1
        child_gen = max(parent_one.generation, parent_two.generation)+1
        newind_one = self.Individual(childchrom_one, child_gen)
        newind_two = self.Individual(childchrom_two, child_gen)

        return newind_one, newind_two

    def mutate(self, individual):
        """
        Mutates each gene of an individual's chromosome in place, based on the
        `mutation_probability`. Mutation applies a Gaussian random value drawn
        from distribution with mean 0 and `stdev = mutation_strength`
        Integer genes get rounded to the nearest int.
        """
        try:
            genetype = self.genetype
            randvars = np.random.random_sample(self.chromlength)
            mutation_value = np.random.normal(0, self.mutation_strength,
                                              self.chromlength)
            newchrom = individual.chromosome +\
                mutation_value*(randvars < self.mutation_probability)
            newchrom = np.clip(newchrom, *self.genebounds)
            if issubclass(individual.chromosome.dtype.type, int):
                newchrom = np.round(newchrom)
            individual.chromosome = newchrom.astype(genetype)
            individual.fitness = None # mark fitness as 'unevaluated'
        except (TypeError, AttributeError):
            exit("ERROR: (Mutation) Invalid chromosome type. "
                 "Must be numpy float or int array.")

    def insert(self, *newinds):
        """
        Inserts the new individuals into the population.
        If the population exceeds the max_population limit, the worst
        individuals from the original population is discarded.

        Assumes the population is evaluated and sorted.
        """
        newpopsize = len(newinds)+len(self.population)
        overflow = newpopsize - self.max_population
        if overflow > 0:
            self.population = self.population[:-overflow]
        self.population.extend(newinds)

    def select(self, method):
        """
        Currently selects individuals randomly for recombination.
        Later it will implement several methods for other selection methods.
        """
        if method == 'rand':
            r1 = r2 = 0
            popsize = len(self.population)
            while r1 == r2:
                r1 = int(round(np.random.random()*(popsize-1)))
                r2 = int(round(np.random.random()*(popsize-1)))
            return self.population[r1], self.population[r2]
        elif method == 'best':
            return self.population[0], self.population[1]
        elif method == 'roulette':
            # TODO: Print to stderr
            print("WARNING: Roulette-wheel selection not yet implemented. "
                  "Falling back to random selection.")
            return self.select('rand')
        else:
            exit("ERROR: Invalid selection method")

    def printprogress(self, gen, bestind, alltime_bestind):
        self.logfile.write(
                "Generation %i\n"
                "Best individual of cur gen:\n"
                "%s\n"
                "Best individual so far:\n"
                "%s\n"
                "---\n" % (gen, bestind, alltime_bestind))

    def optimise(self, num_generations, *optargs):
        """
        Starts the optimisation, i.e., runs the GA loop. Stops after
        `num_generations`. Any subsequent arguments after `num_generations`
        will be passed on to the fitness function as supplied.
        """
        self.sort_population(optargs)
        # copy 0 to initialise the variables
        bestind = copy.deepcopy(self.population[0])
        bestind_age = 0
        # alltime_bestind holds the best individual for when the population is
        # reset
        self.alltime_bestind = copy.deepcopy(self.population[0])
        for gen in range(num_generations):
            p1, p2 = self.select('rand')
            c1, c2 = self.crossover(p1, p2)
            self.mutate(c1)
            self.mutate(c2)
            self.fitnessfunc(c1, *optargs)
            self.fitnessfunc(c2, *optargs)
            self.insert(c1, c2)
            self.sort_population(optargs)
            curbest = copy.deepcopy(self.population[0])
            if curbest.fitness < self.alltime_bestind.fitness:
                self.alltime_bestind = copy.deepcopy(curbest)
            if curbest.fitness == bestind.fitness:
                # could use the actual chromosome instead of just fitness
                bestind_age += 1
            else:
                bestind_age = 0
                bestind = copy.deepcopy(curbest)
            if gen % 100 == 0:  # TODO: Parameterise reporting interval
                self.printprogress(gen, bestind, self.alltime_bestind)
                print("Generation %i\nBest fitness: %f (all time best: %f)\n" % (
                    gen, bestind.fitness, self.alltime_bestind.fitness))
            if bestind_age > 4000:  # arbitrary age limit (TODO: Parameterise)
                # Recreate entire population, except best individual
                # TODO: Also parameterise number of individuals to keep when
                # resetting
                print("Age limit reached. Re-initializing population\n")
                self.logfile.write("Age limit reached. "
                                   "Re-initializing population.\n\n")
                self.init_population(self.max_population, self.chromlength)

        self.printprogress(gen, bestind, self.alltime_bestind)
        print("Final generation\nBest individual fitness:\n%f\n" %
              bestind.fitness)
        self.logfile.close()

    class Individual:

        def __init__(self, chrom, gen):
            self.chromosome = chrom
            self.generation = gen
            self.chromlength = chrom.size
            self.fitness = None # marks fitness as 'unevaluated'

        def __len__(self):
            """Length of Individual is simply length of chromosome"""
            return len(self.chromosome)

        def __repr__(self):
            stringrep = "Chromosome: "
            stringrep += ", ".join(str(c) for c in self.chromosome)
            if self.fitness is not None:
                stringrep += "  Fitness: %f" % self.fitness
            else:
                stringrep += "  <Unevaluated>"
            return stringrep

        def __eq__(self, other):
            """
            Equality operator checks chromosomes for all() equality.
            """
            return np.all(self.chromosome == other.chromosome)

