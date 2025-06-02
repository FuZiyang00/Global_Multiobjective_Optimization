import json
import time
import numpy as np
import pandas as pd
from typing import List

def group_identical_rows_except_last(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Groups rows that are identical except for the last column into separate subdataframes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to process
    
    Returns:
    --------
    List[pd.DataFrame]
        List of subdataframes, each containing rows that are identical except 
        for the last column (only groups with >1 row)
    """
    if df.empty:
        return []
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Add a temporary column to track original indices
    df_copy['_original_index'] = df_copy.index
    
    # Get all columns except the last one (and our temporary index column)
    grouping_columns = list(df.columns[:-1])  # Exclude last column
    
    # Group by all columns except the last column and the temporary index column
    grouped = df_copy.groupby(grouping_columns)
    
    # Filter groups that have more than 1 row and create subdataframes
    subdataframes = []
    for name, group in grouped:
        if len(group) > 1:
            # Remove the temporary column and restore original index
            subdf = group.drop('_original_index', axis=1)
            subdf.index = group['_original_index']
            subdataframes.append(subdf)
    
    return subdataframes

class Filter_Genetic_Algorithm:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = 5
        self.best_individual = None  
        self.best_fitness = 999
        self.fitness_history = []

    
    def initialize_population(self, train_df: pd.DataFrame) -> np.ndarray:
        """Initialize population with random binary chromosomes"""
        individual_size = len(train_df.columns) - 1 # Exclude target column
        population = []
        
        for _ in range(self.population_size):
            # Create random binary chromosome (at least one feature must be selected)
            individual = np.random.randint(0, 2, individual_size)
            # Ensure at least one feature is selected
            if np.sum(individual) == 0:
                individual[np.random.randint(0, individual_size)] = 1
            population.append(individual)
        
        return np.array(population) 


    def inconsistency_rate(self, dataframes: List[pd.DataFrame], total_rows: int) -> float:
        """
        Calculate the inconsistency rate of a list of DataFrames.
        
        Parameters:
        -----------
        dataframes : List[pd.DataFrame]
            List of DataFrames to process
        
        Returns:
        --------
        float
            Inconsistency rate 
        """
        global_inconsistency = 0

        for df in dataframes:
            # Count the occurrences of each target value
            # and calculate the inconsistency
            target_counts = df.TARGET.value_counts()
            inconsistency = int(target_counts.sum()) - int(target_counts.max())
            global_inconsistency += inconsistency
        
        inconsistency_rate = round((global_inconsistency / total_rows ), 2)
        return inconsistency_rate


    def tournament_selection(self, population, fitnesses):
        """Tournament selection - select the best individual from a random subset"""

        if len(population) < self.tournament_size:
            raise ValueError("Population size must be greater than tournament size")
        
        if len(fitnesses) != len(population):
            raise ValueError("Fitnesses must match population size")
        
        # Randomly select individuals for the tournament
        selected = np.random.choice(len(population), self.tournament_size, replace=True)
        best = min(selected, key=lambda i: fitnesses[i])
        return population[best].copy()
    

    def uniform_crossover(self, parent1, parent2):
        """Uniform crossover - each gene has swap_prob chance of being swapped"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < self.crossover_rate:
                child1[i], child2[i] = child2[i], child1[i]
        
        # Ensure at least one feature is selected
        if np.sum(child1) == 0:
            child1[np.random.randint(0, len(child1))] = 1
        if np.sum(child2) == 0:
            child2[np.random.randint(0, len(child2))] = 1
        
        return child1, child2
    

    def bit_flip_mutation(self, individual):
        """Bit flip mutation - randomly flip bits in the chromosome"""

        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least one feature is selected
        if np.sum(mutated) == 0:
            mutated[np.random.randint(0, len(mutated))] = 1
        
        return mutated
    

    def evolve(self, train_df: pd.DataFrame, penalty: float, no_improvement_threshold: int = None):
        """Run the genetic algorithm to evolve the population"""
        population = self.initialize_population(train_df)
        start_time = time.time()  
        full_features_lenght = len(train_df.columns) - 1  # Exclude target column
        log_data = []  # For storing generation logs
        no_improvement_count = 0 
        total_rows = len(train_df)
        
        for generation in range(self.generations):
            fitnesses = []
            
            # Calculate fitness for each individual
            for individual in population:
                selected_indices = np.where(individual == 1)[0]
                selected_features = train_df.columns[:-1][selected_indices]

                # append target column for consistency
                selected_df = train_df[selected_features.tolist() + ['TARGET']]
                sub_df_list = group_identical_rows_except_last(selected_df)
                inconsistency_rate = self.inconsistency_rate(sub_df_list, total_rows)

                # fitness score
                fitness_score = inconsistency_rate + penalty * (len(selected_features) / full_features_lenght)
                fitnesses.append(fitness_score)
            
            
            # Update best individual
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                print(f"Generation {generation}: New best new individual with fitness = {fitnesses[best_idx]:.4f}")
                new_best_fitness = fitnesses[best_idx]
                self.best_individual = population[best_idx]
                self.best_fitness = new_best_fitness
                improvement = True
                no_improvement_count = 0
            else:
                print(f"Generation {generation}: No improvement")
                improvement = False
                no_improvement_count += 1
            
            generation_features = train_df.columns[:-1][np.where(self.best_individual == 1)[0]]
            selected_features_length = len(generation_features)
            feature_delta = full_features_lenght - selected_features_length
            best_binary = ''.join(str(int(bit)) for bit in self.best_individual)

            # Log data for this generation
            log_data.append({
                "generation": generation,
                "best_individual": best_binary,
                "fitness": float(self.best_fitness),
                "feature_delta": int(feature_delta),
                "improvement": improvement
            })
            
            if no_improvement_threshold is not None:
                if no_improvement_count >= no_improvement_threshold:
                    print(f"No improvement for {no_improvement_threshold} generations, stopping evolution.")
                    print(f"reached 'convergence' after {generation} generations")
                    with open("filter_evolution_earlystop_log.json", "w") as f:
                        json.dump(log_data, f, indent=4)
                    return self.best_individual, time.time() - start_time
            
            
            if generation % 10 == 0:
                selected_indices = np.where(self.best_individual == 1)[0]
                generation_features = train_df.columns[:-1][selected_indices]
                selected_features_lenght = len(generation_features)
                print(f"Generation {generation}: Best={self.best_fitness:.4f}, \n Difference with full feature set={full_features_lenght - selected_features_lenght}")
            
            if generation == self.generations -1:
                end_time = time.time()
                elapsed_time = end_time - start_time
                with open("filter_evolution_log.json", "w") as f:
                    json.dump(log_data, f, indent=4)
                return self.best_individual, elapsed_time
            
            # selection 
            selected_population = []
            for i in range(self.population_size // 2):
                # Select two parents using tournament selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                selected_population.extend([parent1, parent2])

            # crossover & mutation
            next_generation = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1] if i + 1 < len(selected_population) else selected_population[0]
                
                # uniform crossover 
                child1, child2 = self.uniform_crossover(parent1, parent2)

                # bit flip mutation
                child1 = self.bit_flip_mutation(child1)
                child2 = self.bit_flip_mutation(child2)
                
                next_generation.extend([child1, child2])
            
            population = np.array(next_generation)