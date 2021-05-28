# Import libraries:
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
def subtract(x): return np.subtract(*x)
def divide(x): return np.divide(*x)


class GeneticAlgorithm:
    """
    Main class for discrete encoding GA.

    Attributes
    ----------
    parameters: list
        fixed parameters for optimization
    population_size: int
        number of individuals in each generation
    population_history: list
        tracking of populations' evolution
    fitness_history: list
        tracking of fitness' evolution
    """

    decode = {0: np.sum,
              1: np.prod,
              2: subtract,
              3: divide}

    decode_str = {0: '+',
                  1: '*',
                  2: '-',
                  3: '/'}

    def __init__(self,
                 parameters: list,
                 population_size: int,
                 target: float):

        self.parameters = parameters
        self.population_size = population_size
        self.population_history = []
        self.fitness_history = []
        self.target = target

    @property
    def population(self):
        """Returns population for current iteration."""
        return self._population

    @population.setter
    def population(self, x):
        """Sets population and calculates fitness for new population."""
        self._population = x
        self.population_history.append(x)
        self.fitness, self.values = self.get_fitness()

    @property
    def fitness(self):
        """Return fitness for current iteration."""
        return self._fitness

    @fitness.setter
    def fitness(self, x):
        """Set fitness for new population."""
        self._fitness = x
        self.fitness_history.append(x)

    def get_fitness(self,
                    population: np.array = None) -> tuple:
        """Calculate fitness and value for new population.

        Parameters
        ----------
        population: array
            individuals for which to calculate fitness

        Returns
        -------
        fitness: array
            fitness score
        value: array
            objective function values
        """
        if population is None:
            population = self.population
        fitness = []
        values = []
        for individual in population:
            total = self.parameters[0]
            for param, gene in enumerate(individual):
                total = self.decode[gene](np.array([total,
                                                    self.parameters[param+1]]))
            values.append(total)
            fitness.append(np.round(abs(self.target - total), 2))
        return np.array(fitness), np.array(values)

    def selection(self,
                  num_parents: int = 4,
                  epsilon: float = 1e-5,
                  method: str = 'proportional',
                  replacement: bool = False) -> np.array:
        """Apply selection operator.

        Parameters
        ----------
        method: str
            choose between 'proportional' - invariant to scale
            and 'smooth' - invariant to scale and translation
        """
        assert num_parents % 2 == 0

        if method == 'proportional':
            crossover_probability = 1/(self.fitness+epsilon)
            crossover_probability /= np.sum(crossover_probability)
            individuals_to_cross = np.random.choice(range(len(crossover_probability)),
                                                    size=num_parents,
                                                    replacement=False,
                                                    p=crossover_probability)
        elif method == 'smoothed':
            crossover_probability = (1+np.argsort(self.fitness))\
                / (len(self.fitness)*(len(self.fitness)+1)/2)
            individuals_to_cross = np.random.choice(range(len(crossover_probability)),
                                                    size=num_parents,
                                                    replace=replacement,
                                                    p=crossover_probability)

        return individuals_to_cross

    def crossover(self,
                  parents: list,
                  method: str = 'uniform') -> np.array:
        """Apply crossover operator.

        Parameters
        ----------
        parents: list(int)
            index of individuals to crossover

        Returns
        -------
        children: np.array
            new individuals (offspring)
        """
        children = []
        if method == 'uniform':
            for parent1_id, parent2_id in zip(parents[::2], parents[1::2]):
                parent1 = self.population[parent1_id]
                parent2 = self.population[parent2_id]
                crossover_mask = np.random.randint(2,
                                                   size=self.population.shape[1])
                mask1, mask2 = [np.where(i == crossover_mask)[0]
                                for i in (0, 1)]
                child1 = np.zeros(self.population.shape[1])
                child2 = np.zeros(self.population.shape[1])
                child1[mask1] = parent1[mask1]
                child1[mask2] = parent2[mask2]
                child2[mask2] = parent1[mask2]
                child2[mask1] = parent2[mask1]
                children.append(np.vstack([child1, child2]))
        children = np.array(children).reshape(len(parents),
                                              self.population.shape[1])
        return children

    def mutation(self,
                 subpopulation: np.array,
                 probability_mutation: float = 0.05,
                 method='gene') -> np.array:
        """Apply mutation operator.

        Parameters
        ----------
        probability_mutation: float
            probability of mutation

        Returns
        -------
        subpopulation: np.array
            Mutated population
        """
        if method == 'gene':
            for individual in subpopulation:
                mutation_idx = np.where(np.random.uniform(
                    size=len(self.parameters)-1) >= 1-probability_mutation)[0]
                if len(mutation_idx) > 0:
                    individual[mutation_idx] = np.random.randint(
                        np.max(list(self.decode)), size=len(mutation_idx))
        return subpopulation

    def replacement(self,
                    subpopulation: np.array,
                    method='SteadyState') -> np.array:
        """Apply replacement operator.

        Parameters
        ----------
        method: str
            Choose among 'SteadyState' - replace a fixed % of old population
            or 'Elitist' - keep fittest from old and new population
        """
        if method == 'elitist':
            newfitness = self.get_fitness(population=subpopulation)[0]
            allpopulation = np.vstack([self.population, subpopulation])
            allfitness = np.hstack([self.fitness, newfitness])
            best_fitted = np.argsort(allfitness)[:self.population.shape[0]]
            newpopulation = allpopulation[best_fitted]
        return newpopulation

    def search(self, iterations=100,
               simulations=20,
               p_mutation=0.05,
               num_parents=4,
               max_gene_convergence=0.95,
               selection_method='smoothed',
               crossover_method='uniform',
               mutation_method='gene',
               replacement_method='elitist') -> None:
        """Search and optimize fitness function."""
        fittest_fitness = []
        fittest_individual = []
        best_fitness = []
        global_best_fitness = 1000

        for simulation in range(0, simulations):
            for iteration in range(0, iterations):
                if iteration == 0:
                    # Initialize population and calculate fitness:
                    self.population = np.random.randint(4,
                                                        size=(self.population_size, len(self.parameters)-1))
                else:
                    # Selection
                    individuals_to_cross = self.selection(num_parents,
                                                          method=selection_method)
                    # Crossover
                    children = self.crossover(parents=individuals_to_cross,
                                              method=crossover_method)
                    # Mutation
                    children = self.mutation(children,
                                             probability_mutation=p_mutation,
                                             method=mutation_method)
                    # Replacement
                    newpopulation = self.replacement(subpopulation=children,
                                                     method=replacement_method)
                    # Set new population (and calculate new fitness)
                    self.population = newpopulation

                # Best fitness in iteration:
                best_fitness_iteration = np.min(self.fitness)
                fittest_fitness.append([simulation,
                                        iteration,
                                        best_fitness_iteration])
                fittest_individual.append([simulation,
                                           iteration,
                                           self.population[np.argmin(self.fitness)]])

                # Convergence (if max_gene_convergence, break this simulation)
                sorted_population = np.sort(self.population, axis=0)
                different_values = (sorted_population[1:, :] !=
                                    sorted_population[:-1, :]).sum(axis=0)+1
                convergence = np.mean(different_values == 1)
                if convergence >= max_gene_convergence:
                    break

                # Updates minimum fitness value if it decreases
                if(best_fitness_iteration < global_best_fitness):
                    global_best_fitness = best_fitness_iteration
                    best_fitness.append([simulation,
                                         iteration,
                                         best_fitness_iteration])
                    if best_fitness_iteration == 0:
                        return
            self.fittest_fitness = fittest_fitness
            self.fittest_individual = fittest_individual
            self.best_fitness = best_fitness
        return

    def result_plot(self, interactive=False) -> plt.plot:
        """Plot best fitness of every iteration."""
        best_fitness_df = pd.DataFrame(self.fittest_fitness,
                                       columns=['simulation', 'iteration', 'bestfitness'])
        all_individual_df = pd.DataFrame(self.fittest_individual,
                                         columns=['simulation', 'iteration', 'individual'])

        absolute_best = all_individual_df[(all_individual_df.simulation ==
                                           self.best_fitness[-1][0]) &
                                          (all_individual_df.iteration ==
                                           self.best_fitness[-1][1])]

        display(absolute_best)
        absolute_best_fitness, absolute_best_values = self.get_fitness(
            absolute_best.individual.values)
        print(
            f'Fitness:{absolute_best_fitness[0]} \t Value: {absolute_best_values[0]}')

        if interactive:
            y_max = best_fitness_df.bestfitness.max(),
            y_min = best_fitness_df.bestfitness.min()
            fig = px.line(best_fitness_df,
                          x='iteration',
                          y='bestfitness',
                          animation_frame="simulation",
                          animation_group="iteration",
                          range_x=[0, 100],
                          range_y=[y_min, y_max],
                          title='Best fitness')
        else:
            fig = px.line(best_fitness_df,
                          x='iteration',
                          y='bestfitness',
                          color='simulation')
        fig.show()

    def convergence_plot(self, interactive=False) -> plt.plot:
        """ Plot convergence metrics.

            Display plots for offline measure & online measure
            as seen in ([1] DeJong K., 1975)
        """
        def offline_measure(df):
            return pd.Series(df['fitness'].cumsum().values /
                             (df['iteration'].values+1))

        def online_measure(df):
            return pd.Series(df['mean_fitness'].cumsum().values /
                             (df['iteration'].values+1))

        return NotImplementedError

    def __str__(self):
        """String representation of last population stored."""
        text = []
        for i, individual in enumerate(self.population):
            total = str(self.parameters[0])
            for param, gene in enumerate(individual):
                total += f'{self.decode_str[gene]}{self.parameters[param+1]}'
            total += f'={round(self.values[i],2)} \t - fitness:{round(self.fitness[i],2)}'
            text.append(total)
        return ", \n".join(text)
