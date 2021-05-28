# Import libraries:
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class GeneticAlgorithm:
    """
    Main class for real encoding GA.

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
    minimum_values: list
        lower bound value for every gene 
        [lower_bound_gene1, ..., lower_bound_geneN]
    maximum_values: list
        upper bound value for every gene 
        [upper_bound_gene1, ..., upper_bound_geneN]
    """

    def __init__(self,
                 parameters: list,
                 population_size: int,
                 minimum_values: list,
                 maximum_values: list):

        self.parameters = parameters
        self.population_size = population_size
        self.population_history = []
        self.fitness_history = []
        self.minimum_values = minimum_values
        self.maximum_values = maximum_values

    @property
    def population(self):
        """Return population for current iteration."""
        return self._population

    @population.setter
    def population(self, x):
        """Set population and calculates fitness for new population.
        """
        self._population = x
        self.population_history.append(x)
        self.fitness = self.get_fitness()

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
                    population: np.array = None) -> np.array:
        """Calculate fitness and value for new population.

        Parameters:
        ----------
        population: array
            individuals for which to calculate fitness

        Returns
        -------
        fitness: array
            fitness score
        """
        if population is None:
            population = self.population
        fitness = []
        for individual in population:
            # TODO:
            # Fill with custom fitness function
            raise NotImplementedError
        return np.array(fitness)

    def selection(self,
                  num_parents: int = 4,
                  epsilon: float = 1e-5,
                  method: str = 'proportional',
                  replacement: bool = False) -> np.array:
        """ Apply selection operator.

        Parameters:
        ----------
        num_parents: int
            number of parents to include in mating pool.
            Also number of children to output.
        method: str
            choose between 'proportional' - invariant to scale
            and 'smooth' - invariant to scale and translation
        epsilon: float
            constant
        replacement: bool, optional
            optional to choose with or without replacement

        Returns
        -------
        individuals_to_cross: np.array
            index of individuals to cross
        """
        assert num_parents % 2 == 0

        if method == 'proportional':
            crossover_probability = self.fitness/np.sum(crossover_probability)
            individuals_to_cross = np.random.choice(range(len(crossover_probability)),
                                                    size=num_parents,
                                                    replacement=False,
                                                    p=crossover_probability)
        elif method == 'smoothed':
            # Highest fitness individual, gets chosen more often:
            crossover_probability = (len(self.fitness)-(np.argsort(-self.fitness)))\
                / (len(self.fitness)*(len(self.fitness)+1)/2)
            individuals_to_cross = np.random.choice(range(len(crossover_probability)),
                                                    size=num_parents,
                                                    replace=replacement,
                                                    p=crossover_probability)
        return individuals_to_cross

    def crossover(self, parents: list, method: str = 'BLX') -> np.array:
        """ Apply crossover operator.

        Parameters
        ----------
        parents: list
            index of individuals to crossover
        method: str
            crossover operator.
            Options - 'BLX' (blend crossover, 
                             within parents interval)

        Returns
        -------
        children: np.array
            new individuals (offspring)
        """
        children = []
        if method == 'BLX':
            for parent1_id, parent2_id in zip(parents[::2], parents[1::2]):
                parent1 = self.population[parent1_id]
                parent2 = self.population[parent2_id]
                stacked = np.vstack([parent1, parent2])
                min_ab = np.min(stacked, axis=0)
                max_ab = np.max(stacked, axis=0)
                child1 = np.random.uniform(min_ab, max_ab)
                child2 = max_ab-(child1-min_ab)
                children.append(np.vstack([child1, child2]))
        children = np.array(children).reshape(len(parents),
                                              self.population.shape[1])
        return children

    def mutation(self, subpopulation: np.array,
                 probability_mutation: float = 0.05,
                 method: str = 'gene') -> np.array:
        """Apply mutation operator.

        Parameters
        ----------
        probability_mutation: float
            probability of mutation

        Returns
        -------
        subpopulation: np.array
            population with mutations
        """
        if method == 'gene':
            for individual in subpopulation:
                mutation_idx = np.where(np.random.uniform(
                    size=self.population.shape[1]) >= 1-probability_mutation)[0]
                if len(mutation_idx) > 0:
                    individual[mutation_idx] = np.random.uniform(
                        self.minimum_values[mutation_idx],
                        self.maximum_values[mutation_idx])
        return subpopulation

    def replacement(self,
                    subpopulation: np.array,
                    method='SteadyState') -> np.array:
        """Apply replacement operator.

        Parameters
        ----------
        subpopulation: np.array
            subsample of population.
        method: str
            Choose 'SteadyState' - replace a fixed % of old population
            or 'Elitist' - keep fittest from old and new population
        """
        if method == 'elitist':
            newfitness = self.get_fitness(population=subpopulation)[0]
            allpopulation = np.vstack([self.population,
                                       subpopulation])
            allfitness = np.hstack([self.fitness,
                                    newfitness])
            best_fitted = np.argsort(-allfitness)[:self.population.shape[0]]
            newpopulation = allpopulation[best_fitted]
        elif method == 'SteadyState':
            raise NotImplementedError
        return newpopulation

    def search(self,
               iterations: int = 100,
               simulations: int = 20,
               p_mutation: float = 0.05,
               num_parents: int = 4,
               max_gene_convergence: float = 0.95,
               selection_method: str = 'smoothed',
               crossover_method: str = 'BLX',
               mutation_method: str = 'gene',
               replacement_method: str = 'elitist') -> None:
        """Search and optimize fitness function.

        Parameters
        ----------
        iterations: int, optional
            number of maximum generations for a
            given simulation
        simulations: int, optional
            number of times to initialize GA
        p_mutation: float, optional
            probability of an individual mutating
        num_parents: int, optional
            number of parents to include in a 
            mating pool
        max_gene_convergence: float, optional
            stopping criteria for generation
        selection_method: str, optional
            selection operator
            Options - 'smoothed', 'proportional'
        crossover_method: str, optional
            crossover operator. Options - 'BLX'
        mutation_method: str, optional
            mutation operator. Options - 'gene'
        replacement_method: str, optional
            replacement operator.
            Options - 'elitist', 'SteadyState'

        Returns
        -------
        population_history:np.array
            history of populations' evolution
        fitness_history:np.array
            history of fitness' evolution
        """
        fittest_fitness = []
        fittest_individual = []
        best_fitness = []
        global_best_fitness = -100000
        self.simulations = simulations
        self.iterations = iterations
        for simulation in range(0, simulations):
            for iteration in range(0, iterations):
                if iteration == 0:
                    # Initialize population and calculate fitness:
                    self.population = np.random.uniform(self.minimum_values,
                                                        self.maximum_values,
                                                        size=(self.population_size,
                                                              len(self.minimum_values)))
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

                # Best fitness in iteration, choose Max or Min:
                best_fitness_iteration = np.max(self.fitness)
                fittest_fitness.append([simulation,
                                        iteration,
                                        best_fitness_iteration])
                fittest_individual.append([simulation,
                                           iteration,
                                           self.population[np.argmax(self.fitness)]])

                # Convergence (if max_gene_convergence, break this simulation)
                sorted_population = np.sort(self.population, axis=0)
                different_values = (
                    sorted_population[1:, :] != sorted_population[:-1, :]).sum(axis=0)+1
                convergence = np.mean(different_values == 1)
                if convergence >= max_gene_convergence:
                    break

                # Updates max fitness value if it increases
                if(best_fitness_iteration > global_best_fitness):
                    global_best_fitness = best_fitness_iteration
                    best_fitness.append([simulation,
                                         iteration,
                                         best_fitness_iteration])

        self.fittest_fitness = fittest_fitness
        self.fittest_individual = fittest_individual
        self.best_fitness = best_fitness
        return self.population_history, self.fitness_history

    def get_best_parameters(self):
        """Retrieve best parameters found."""
        best_sim, best_it, best_fit = self.best_fitness[-1]
        best_individual_array = np.array(self.fittest_individual)
        best_individual_array = best_individual_array.reshape(self.simulations,
                                                              self.iterations,
                                                              -1)
        best_individual = best_individual_per_it[best_sim][best_it][-1]
        return best_individual

    def result_plot(self,
                    interactive: bool = False) -> plt.plot:
        """Plot best fitness of every iteration.

        Parameters
        ----------
        interactive: bool
            choose if plot is interactive or not
        """
        best_fitness_df = pd.DataFrame(self.fittest_fitness,
                                       columns=['simulation', 'iteration', 'bestfitness'])
        all_individual_df = pd.DataFrame(self.fittest_individual,
                                         columns=['simulation', 'iteration', 'individual'])

        absolute_best = all_individual_df[(all_individual_df.simulation ==
                                           self.best_fitness[-1][0]) &
                                          (all_individual_df.iteration ==
                                           self.best_fitness[-1][1])]

        display(absolute_best)
        absolute_best_fitness = self.get_fitness(
            absolute_best.individual.values)
        print(f'Fitness:{absolute_best_fitness[0]}')
        if interactive:
            y_max = best_fitness_df.bestfitness.max()
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

    def convergence_plot(self,
                         interactive: bool = False) -> plt.plot:
        """ Plot convergence metrics.

            Display plots for offline measure & online measure
            as seen in ([1] DeJong K., 1975)

            Parameters
            ----------
            interactive: bool
                choose if plot is interactive or not
        """
        def offline_measure(df):
            return pd.Series(df['fitness'].cumsum().values /
                             (df['iteration'].values+1))

        def online_measure(df):
            return pd.Series(df['mean_fitness'].cumsum().values /
                             (df['iteration'].values+1))

        fittest_fitness_df = pd.DataFrame(self.fittest_fitness,
                                          columns=['simulation',
                                                   'iteration',
                                                   'fitness'])
        offline_measures = fittest_fitness_df.groupby('simulation')\
            .apply(lambda x: offline_measure(x))
        offline_measures = pd.pivot_table(offline_measures, index='simulation')\
            .stack().reset_index()\
            .rename(columns={'level_1': 'iteration', 0: 'm*'})
        offline_measures['iteration'] = fittest_fitness_df['iteration']

        fig = px.line(offline_measures,
                      x='iteration',
                      y='m*',
                      title='Offline measure',
                      color='simulation')
        fig.show()

        fittest_fitness_df['mean_fitness'] = np.mean(self.fitness_history,
                                                     axis=1)
        online_measures = fittest_fitness_df.groupby('simulation')\
            .apply(lambda x: online_measure(x))
        online_measures = pd.pivot_table(online_measures,
                                         index='simulation')\
            .stack().reset_index()\
            .rename(columns={'level_1': 'iteration',
                             0: 'm'})
        online_measures['iteration'] = fittest_fitness_df['iteration']

        fig = px.line(online_measures,
                      x='iteration',
                      y='m',
                      title='Online measure',
                      color='simulation')
        fig.show()
        if interactive:

            # Create figure
            fig = go.Figure()
            # Add traces, one for each slider step
            for step in set(online_measures['simulation']):
                fig.add_trace(
                    go.Scatter(
                        visible=False,
                        line=dict(color="#00CED1", width=3),
                        name="m  - Online measure ",
                        x=online_measures.loc[online_measures['simulation'] == step,
                                              'iteration'],
                        y=online_measures.loc[online_measures['simulation'] == step,
                                              'm']))
                fig.add_trace(
                    go.Scatter(
                        visible=False,
                        line=dict(color="#008888", width=3),
                        name="m* - Offline measure ",
                        x=offline_measures.loc[offline_measures['simulation'] == step,
                                               'iteration'],
                        y=offline_measures.loc[offline_measures['simulation'] == step,
                                               'm*']))
            # Make 1st trace visible
            fig.data[1].visible = True

            # Create and add slider
            steps = []
            for i in range(0, len(fig.data), 2):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)},
                          {"title": f"Offline-online measure - Simulation {i+1}"}])
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][i] = True
                step["args"][0]["visible"][i+1] = True
                steps.append(step)

            sliders = [dict(
                active=10,
                currentvalue={"prefix": "Simulation: "},
                pad={"t": 50},
                steps=steps
            )]

            fig.update_layout(
                sliders=sliders
            )

            fig.show()
        return

    def __str__(self):
        """String representation of last population stored."""
        text = []
        for i, individual in enumerate(self.population):
            total = f'param1={individual[0]},param2={individual[1]},param3=[{individual[2]}]'
            total += f'\t - fitness:{round(self.fitness[i],2)}'
            text.append(total)
        return ", \n".join(text)
