import re
import random
import xcs
import math


# Define the regular expression to be learned
regex = r'^[01]+0$'

# Define the XCS algorithm parameters
xcs_params = {
    'max_population': 1000,
    'crossover_probability': 0.8,
    'mutation_probability': 0.04,
    'subsumption_threshold': 20,
    'ga_threshold': 25,
    'theta_mna': 0.1,
    'theta_ga': 100,
    'theta_as': 0.9,
    'mu': 0.05,
    'epsilon_0': 0.01,
    'alpha': 0.1,
    'gamma': 0.71,
    'n': 10,
    'time': None,
    'is_covering': True
}


import random
import re

class Classifier:
    def init(self, condition=None):
        self.condition = condition or [''] * 10
        self.action = random.choice(['0', '1'])
        self.prediction = random.uniform(0, 1)
        self.error = 0
        self.fitness = 0
        self.experience = 0
    
   
    def does_match(self, string):
        for i, char in enumerate(self.condition):
            if char == '' or char == string[i]:
                continue
            else:
                return False
        return True
    
    def get_subexpressions(self):
        subexpressions = []
        for i in range(len(self.condition)):
            if self.condition[i] in '.|()':
                continue
            if i < len(self.condition) - 1 and self.condition[i+1] == '*':
                subexpressions.append(self.condition[i:i+2])
            else:
                subexpressions.append(self.condition[i])
        return subexpressions
    
    def is_match(self):
        return '' not in self.condition

    def is_reliable(self):
        return self.experience > 20 and self.error < 0.2

    def predict(self, string):
        if self.does_match(string):
            return self.prediction
        else:
            return 0

    def update_fitness(self, accuracy):
        if self.fitness == 0:
            self.fitness = 0.1
        else:
            self.fitness += 0.1 * (accuracy - self.fitness)

    def update_error(self, accuracy):
        if self.error == 0:
            self.error = 0.1
        else:
            self.error += 0.1 * (abs(accuracy - self.prediction) - self.error)

    def update_experience(self):
        self.experience += 1
        
    def set_fitness(self, fitness):
        self.fitness = fitness
        
    def crossover(self, other):
        child1 = Classifier()
        child2 = Classifier()

        for i in range(10):
            if random.random() < 0.5:
                child1.condition[i] = self.condition[i]
                child2.condition[i] = other.condition[i]
            else:
                child1.condition[i] = other.condition[i]
                child2.condition[i] = self.condition[i]

        return child1, child2

    def mutate(self):
        if random.random() < 0.01:
            i = random.randint(0, 9)
            self.condition[i] = random.choice(['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                                                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

    def repr(self):
        return ' '.join(self.condition) + ' -> ' + self.action

class XCSAlgorithm:
    def init(self, params):
        self.params = params
        self.population = []
   
    def generate_training_strings(num_strings, string_length):
        """
        Generate a list of random training strings of length string_length.
        """
        training_strings = []
        for i in range(num_strings):
            training_string = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(string_length))
            training_strings.append(training_string)
        return training_strings
    def show_regex(self):
        # Sort the population by fitness and prediction length
        sorted_population = sorted(self.population, key=lambda c: (c.fitness, -len(c.action)))
        # Generate a list of unique subexpressions
        subexpressions = []
        for classifier in sorted_population:
            for subexpression in classifier.get_subexpressions():
                if subexpression not in subexpressions:
                    subexpressions.append(subexpression)

        # Combine the subexpressions into a single regular expression
        regex = '|'.join(subexpressions)

        return regex
    def run_experiment(self, rounds):
        for i in range(rounds):
            # Generate a random string
            random_string = ''.join([random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                                                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']) for _ in range(10)])

            # Check if the string matches the regex
            if re.match(r'^[a-z]{10}$', random_string):
                target = 1
            else:
                target = 0

            # Train the XCS classifier
            self.train(random_string, target)
        # Sort the classifiers by their fitness
        sorted_classifiers = sorted(self.population, key=lambda c: c.fitness, reverse=True)

        # Find the first classifier that matches the target string
        for classifier in sorted_classifiers:
            if classifier.does_match('0010'):
                return classifier.regex()

        # If no classifier matches the target string, return None
        return None

    def train(self, string, target):
        # Generate a match set
        match_set = [classifier for classifier in self.population if classifier.does_match(string)]

        # Generate a covering classifier if the match set is empty
        if not match_set:
            new_classifier = Classifier(string)
            new_classifier.set_fitness(target)
            self.population.append(new_classifier)
        else:
            # Generate a prediction array for the match set
            prediction_array = [classifier.predict(string) for classifier in match_set]

            # Calculate the accuracy of the prediction array
            accuracy = sum(prediction_array) / len(prediction_array)

            # Generate a correct set
            correct_set = [classifier for classifier in match_set if classifier.predict(string) == target]

            # Calculate the fitness of the classifiers in the correct set
            for classifier in correct_set:
                classifier.update_fitness(accuracy)

            # Generate a covering classifier if the correct set is empty
            if not correct_set:
                new_classifier = Classifier(string)
                new_classifier.set_fitness(target)
                self.population.append(new_classifier)
            else:
                # Select the classifier with the highest prediction value as the action
                q_values = {classifier: classifier.predict(string) for classifier in correct_set}
                max_q = max(q_values.values())
                action = random.choice([classifier for classifier, q_value in q_values.items() if q_value == max_q])

                # Update the action's prediction value
                action.update_prediction(accuracy)

                # Apply the genetic algorithm if the population size exceeds the maximum population size
                if len(self.population) > self.params['max_population']:
                    self.genetic_algorithm()
                    
    def genetic_algorithm(self):
        # Generate the parent set
        parent_set = [classifier for classifier in self.population if classifier.is_match() and classifier.is_reliable()]

        # Generate the offspring set
        offspring_set = []
        while len(offspring_set) < self.params['max_population']:
            parent1 = random.choice(parent_set)
            parent2 = random.choice(parent_set)
            # Crossover the parents to generate an offspring
            if random.random() < self.params['crossover_probability']:
                offspring = parent1.crossover(parent2)
            else:
                offspring = parent1.copy()

            # Mutate the offspring
            if random.random() < self.params['mutation_probability']:
                offspring.mutate()

            # Add the offspring to the population
            offspring_set.append(offspring)

        # Update the population using the offspring set
        for offspring in offspring_set:
            # Check if the offspring can subsume any classifiers in the parent set
            subsuming_classifiers = [classifier for classifier in parent_set if classifier.does_subsume(offspring)]
            if subsuming_classifiers:
                # Select the most general classifier that subsumes the offspring
                subsuming_classifiers.sort(key=lambda x: x.get_generality())
                subsuming_classifiers[0].subsume(offspring)
            else:
                # Add the offspring to the population if the population size is below the maximum
                if len(self.population) < self.params['max_population']:
                    self.population.append(offspring)
                else:
                    # Select a classifier to delete from the population
                    fitness_sum = sum([classifier.get_fitness() for classifier in self.population])
                    deletion_probabilities = [classifier.get_fitness() / fitness_sum for classifier in self.population]
                    deletion_index = random.choices(range(len(self.population)), weights=deletion_probabilities)[0]
                    del self.population[deletion_index]

                    # Add the offspring to the population
                    self.population.append(offspring)
    
xcs = XCSAlgorithm(xcs_params)

    # Run the experiment for 1000 rounds
xcs.run_experiment(1000)

            # Show the regular expression that was learned
print(f"Learned regex: {xcs.show_regex()}")
