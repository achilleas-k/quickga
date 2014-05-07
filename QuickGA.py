'''
TODO:
    - Errors shouldn't call exit.
    - Parameterise everything.
'''
from sys import exit, float_info
import copy
import numpy as np
import warnings
import multiprocessing

class GA:

    def __init__(self, max_population, chromlength,\
            mutation_probability=1, mutation_strength=0.1,\
            selection_percentage=0.2, init_pop=None, outputfile='quickga.log'):
        self.max_population = max_population
        self.mutation_probability = mutation_probability
        self.mutation_strength = mutation_strength
        self.selection_percentage = selection_percentage
        self.chromlength = chromlength
        self.population = init_pop
        self.outputfile = open(outputfile, 'w')
        if self.population is None:
            self.population = []
            self.init_population(self.max_population, chromlength)

    def init_population(self, max_population, chromlength):
        del(self.population)
        self.population = []
        for p in range(max_population):
            randchrom = np.random.random(chromlength)
            newind = self.Individual(randchrom, 0)
            self.population.append(newind)

    def fitnessfunc(self, individual, optargs):
        '''
        Calculates the fitness value of an individual, stores it in the
        individual's `fitness` member and returns it.
        This function is the target function to be optimised
        by the GA.
        '''
        return None

    def evaluate_population(self, optargs):
        '''
        Updates the fitness values of the population, for all individuals
        flagged as unevaluated.
        '''
        for ind in self.population:
            if ind.fitness is None:
                self.fitnessfunc(ind, optargs)

    def sort_population(self, optargs):
        '''
        Sorts the population based on fitness values such that the first
        individual is the best in the population.
        This method first calls the `evaluate_population` method to calculate
        the fitness values of any unevaluated individuals
        '''
        self.evaluate_population(optargs)
        self.population.sort(key=lambda individual: individual.fitness)

    def crossover(self, parent_one, parent_two):
        '''
        Two-point crossover.
        '''
        parchrom_one = copy.deepcopy(parent_one.chromosome)
        parchrom_two = copy.deepcopy(parent_two.chromosome)
        if parent_one.chromlength != parent_two.chromlength:
            exit('ERROR: Chromosome lengths don\'t match. '
                        'Skipping crossover.')

        cutpoint_one = cutpoint_two = 0
        chromlength = parent_one.chromlength

        while cutpoint_one == cutpoint_two:
            cutpoint_one = int(np.floor(np.random.random()*(chromlength-1)))
            cutpoint_two = int(np.floor(np.random.random()*(chromlength-1)))

        cutpoint_one, cutpoint_two = (min(cutpoint_one, cutpoint_two),
                                        max(cutpoint_one, cutpoint_two))
        childchrom_one = np.append(parchrom_one[0:cutpoint_one],
                            parchrom_two[cutpoint_one:cutpoint_two])
        childchrom_one = np.append(childchrom_one, parchrom_one[cutpoint_two:])
        childchrom_two = np.append(parchrom_two[0:cutpoint_one],
                            parchrom_one[cutpoint_one:cutpoint_two])
        childchrom_two = np.append(childchrom_two, parchrom_two[cutpoint_two:])
        if childchrom_one.size != childchrom_two.size:
            exit('WTF')
        if not childchrom_one.size == parchrom_one.size ==\
                childchrom_two.size == parchrom_two.size:
                    exit('ERROR: Chromosome lengths changed during crossover')

        '''
        Turn childchroms into individuals
        Childrens' generation will be youngest parent's gen+1
        '''
        last_gen = max(parent_one.generation, parent_two.generation)
        newind_one = self.Individual(childchrom_one, last_gen+1)
        newind_two = self.Individual(childchrom_two, last_gen+1)

        return newind_one, newind_two

    def mutate(self, individual):
        '''
        Mutates an individual's chromosomes in place.
        '''
        try:
            if individual.chromosome.dtype is np.dtype(np.float):
                individual.chromosome += \
                    np.random.normal(0, self.mutation_strength,\
                        individual.chromlength)
            individual.fitness = None # mark fitness as 'unevaluated'
        except (TypeError, AttributeError):
            exit('ERROR: (Mutation) Invalid chromosome type. '
                            'Must be numpy float or int array.')

    def insert(self, newinds):
        '''
        Inserts the new individuals into the population.
        If the population exceeds the max_population limit, the worst
        individuals from the original population is discarded.

        Assumes the population is evaluated and sorted.
        '''
        newpopsize = len(newinds)+len(self.population)
        overflow = newpopsize - self.max_population
        if overflow <= 0:
            self.population.append(newinds)
        else:
            self.population = self.population[:-overflow] + newinds

    def select(self, method):
        '''
        Currently selects individuals randomly for recombination.
        Later it will implement several methods for other selection methods.
        '''
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
            print('WARNING: Roulette-wheel selection not yet implemented. \
Falling back to random selection.')
            return self.select('rand')
        else:
            exit('ERROR: Invalid selection method')

    def saveprogress(self, gen, bestind, alltime_bestind):
        self.outputfile.write(
                'Generation %i\n'
                'Best individual of cur gen:\n'
                '%s\n'
                'Best individual so far:\n'
                '%s\n'
                '---\n' % (gen, bestind, alltime_bestind))

    def optimise(self, num_generations, *optargs):
        self.sort_population(optargs)
        # it is not significant which ind we use for best and alltime best
        # we just copy 0 to initialise the variables
        bestind = copy.deepcopy(self.population[0])
        bestind_age = 0
        # alltime_bestind holds the best individual for when the population is reshuffled
        alltime_bestind = copy.deepcopy(self.population[0])
        for gen in range(num_generations):
            p1, p2 = self.select('rand')
            c1, c2 = self.crossover(p1, p2)
            self.mutate(c1)
            self.mutate(c2)
            self.fitnessfunc(c1, optargs)
            self.fitnessfunc(c2, optargs)
            self.insert([c1, c2])
            self.sort_population(optargs)
            curbest = copy.deepcopy(self.population[0])
            if curbest.fitness < alltime_bestind.fitness:
                alltime_bestind = copy.deepcopy(curbest)
            if curbest.fitness == bestind.fitness:
                # could use the actual chromosome instead of just fitness
                bestind_age += 1
            else:
                bestind_age = 0
                bestind = copy.deepcopy(curbest)
            if gen % 100 == 0:
                self.saveprogress(gen, bestind, alltime_bestind)
                print('Generation %i\nBest fitness: %f (all time best: %f)\n' % (
                    gen, bestind.fitness, alltime_bestind.fitness))
            if bestind_age > 4000: #arbitrary age limit
                '''
                Recreate entire population, except best individual.
                '''
                print('Age limit reached. Re-initializing population\n')
                self.outputfile.write('Age limit reached. '
                        'Re-initializing population.\n\n')
                self.init_population(self.max_population, self.chromlength)

        self.saveprogress(gen, bestind, alltime_bestind)
        print('Final generation\nBest individual:\n%s\n' % (
            bestind))
        self.outputfile.close()

    class Individual:

        def __init__(self, chrom, gen):
            self.chromosome = chrom
            self.generation = gen
            self.chromlength = chrom.size
            self.fitness = None # marks fitness as 'unevaluated'

        def __len__(self):
            return len(self.chromosome)

        def __repr__(self):
            stringrep = 'Chromosome:\n'
            for c in self.chromosome:
                stringrep += '%f, ' % c
            stringrep = stringrep[:-2]
            stringrep += '\nFitness: %f\n' % self.fitness
            return stringrep

        def __eq__(self, other):
            '''
            Equality operator checks chromosomes for all() equality.
            '''
            return np.all(self.chromosome == other.chromosome)



