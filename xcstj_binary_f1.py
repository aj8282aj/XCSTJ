import random
import re
from collections import defaultdict
import numpy as np
class XCSBinaryRegex:
    def __init__(self, input_size, population_size, crossover_rate, mutation_rate, theta_ga, theta_as, theta_exp, p_sharp, mu, nu, chi, initial_error, subsumption_threshold, tournament_size):
        self.input_size = input_size
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.theta_ga = theta_ga
        self.theta_as = theta_as
        self.theta_exp = theta_exp
        self.p_sharp = p_sharp
        self.mu = mu
        self.nu = nu
        self.chi = chi
        self.initial_error = initial_error
        self.subsumption_threshold = subsumption_threshold
        self.tournament_size = tournament_size

        self.population = []
        self.action_set = []
        self.time_stamp = 0
        self.average_size = 0

    def initialize_population(self):
        for i in range(self.population_size):
            condition = "".join(random.choice("01#") for _ in range(self.input_size))
            action = self.generate_random_regex()
            self.population.append({"condition": condition, "action": action, "error": self.initial_error, "fitness": 0, "numerosity": 1, "experience": 0, "last_time_stamp": 0, "action_set_size": 0, "average_size": len(action)})
        self.average_size = self.population_size

    def generate_random_regex(self):
        regex = ""
        n = random.randint(1, self.input_size)
        for i in range(n):
            regex += random.choice(["0", "1"])
        return regex

    def select_action(self, state):
        matching_classifiers = []
        for classifier in self.population:
            if re.match(classifier["condition"], state):
                matching_classifiers.append(classifier)

        if not matching_classifiers:
            return None

        action_dict = defaultdict(int)
        total_fitness = 0

        for classifier in matching_classifiers:
            total_fitness += classifier["fitness"]
            action_dict[classifier["action"]] += classifier["numerosity"]

        p_exploration = self.p_sharp / (self.p_sharp + len(action_dict))
        if random.random() < p_exploration:
            action = random.choice(list(action_dict.keys()))
        else:
            action = max(action_dict, key=action_dict.get)

        return action
    
    def calculate_accuracy(self, action, state):
        regex = action.replace("0", "\\d").replace("1", "[01]")
        pattern = re.compile(regex)
        total_count = 0
        correct_count = 0

        for i in range(2 ** self.input_size):
            s = format(i, f"0{self.input_size}b")
            if re.match(pattern, s):
                total_count += 1
                if s == state:
                    correct_count += 1

        if total_count == 0:
            return 0

        return correct_count / total_count

    def set_fitness(self, state, reward):
        matching_classifiers = []
        for classifier in self.population:
            if re.match(classifier["condition"], state):
                matching_classifiers.append(classifier)

        if not matching_classifiers:
            return

        accuracy_sum = 0
        for classifier in matching_classifiers:
            accuracy_sum += self.calculate_accuracy(classifier["action"], state)

        for classifier in matching_classifiers:
            accuracy = self.calculate_accuracy(classifier["action"], state)
            accuracy_fraction = accuracy / accuracy_sum
            classifier["fitness"] += self.nu * (accuracy_fraction * (reward - classifier["fitness"]))
            classifier["error"] += self.nu * (accuracy - classifier["error"])

        def calculate_accuracy(self, action, state):
            """
            Calculate accuracy of the action with respect to the given state.

            :param action: A string representing a regular expression.
            :param state: A string representing a binary input.
            :return: A float representing the accuracy of the action on the given state.
            """
            if not state:
                return 0.0

            # Compile the action into a regular expression pattern.
            pattern = re.compile(action)

            # Match the regular expression pattern against the state.
            match = pattern.fullmatch(state)

            # Calculate the accuracy of the action as the ratio of the length of the match to the length of the state.
            return len(match.group(0)) / len(state) if match else 0.0
    def apply_crossover(self, parent_1, parent_2):
            if random.random() < self.crossover_rate:
                crosspoint_1 = random.randint(0, self.input_size-1)
                crosspoint_2 = random.randint(crosspoint_1+1, self.input_size)

                child_1 = parent_1[:crosspoint_1] + parent_2[crosspoint_1:crosspoint_2] + parent_1[crosspoint_2:]
                child_2 = parent_2[:crosspoint_1] + parent_1[crosspoint_1:crosspoint_2] + parent_2[crosspoint_2:]

                return child_1, child_2

            return parent_1, parent_2

    def apply_mutation(self, string):
        mutated_string = ""
        for bit in string:
            if random.random() < self.mutation_rate:
                mutated_string += "1" if bit == "0" else "0"
            else:
                mutated_string += bit

        return mutated_string
    def calculate_reward(self, state, action):
        regex = action.replace("0", "\\d").replace("1", "[01]")
        pattern = re.compile(regex)

        if pattern.fullmatch(state):
            return 100
        else:
            return 0
    def update_action_set(self, state):
        action_set = []
        matching_classifiers = []
        for classifier in self.population:
            if re.match(classifier["condition"], state):
                matching_classifiers.append(classifier)
                if classifier["action"] not in action_set:
                    action_set.append(classifier["action"])
                classifier["experience"] += 1
                classifier["action_set_size"] = len(action_set)
        return action_set, matching_classifiers
    
    def run_experiment(self, num_iterations):
        self.initialize_population()

        for i in range(num_iterations):
            print(f"Iteration {i+1}")
            for j in range(2**self.input_size):
                state = format(j, f"0{self.input_size}b")

                action = self.select_action(state)

                if action is None:
                    action = self.generate_random_regex()
                    self.action_set.append({"condition": state, "action": action, "experience": 0, "last_time_stamp": self.time_stamp})

                reward = self.calculate_reward(state, action)

                self.set_fitness(state, reward)

                if self.time_stamp - max(cl["last_time_stamp"] for cl in self.action_set if cl["action"] == action) > self.theta_exp:
                    action_set_numerosity = sum(cl["numerosity"] for cl in self.action_set if cl["action"] == action)
                    self.action_set = [cl for cl in self.action_set if cl["action"] != action]
                    for cl in self.population:
                        if cl["action"] == action:
                            cl["numerosity"] += action_set_numerosity
                            cl["average_size"] += (len(action) - cl["average_size"]) / cl["numerosity"]

                if reward == 1:
                    self.update_action_set(action)

                if self.time_stamp - max(cl["last_time_stamp"] for cl in self.population if cl["action"] == action) > self.theta_ga:
                    parent_1 = self.select_offspring(action)
                    parent_2 = self.select_offspring(action)

                    child_1, child_2 = self.apply_crossover(parent_1["condition"], parent_2["condition"])
                    child_1 = self.apply_mutation(child_1)
                    child_2 = self.apply_mutation(child_2)

                    self.insert_classifier(child_1, action)
                    self.insert_classifier(child_2, action)

                self.delete_from_population()

                self.time_stamp += 1

            self.run_tournament(i)
    def insert_classifier(self, classifier):
        if len(self.population) >= self.population_size:
            least_fit_classifier = min(self.population, key=lambda cl: cl["fitness"] * cl["numerosity"])
            self.delete_classifier(least_fit_classifier)

        self.population.append(classifier)

    def subsume_classifier(self, classifier, state):
            subsumer_found = False
            for subsumer in self.population:
                if subsumer["experience"] > self.theta_exp and subsumer["action"] == classifier["action"] and subsumer["condition"] in state and subsumer["condition"] != classifier["condition"]:
                    subsumer_found = True
                    subsumer["numerosity"] += 1
                    if classifier["experience"] > subsumer["experience"]:
                        subsumer["experience"] = classifier["experience"]
                    break

            if not subsumer_found:
                self.insert_classifier(classifier["condition"], classifier["action"])
   
    def delete_classifier(self):
        total_numerosity = sum(clf["numerosity"] for clf in self.population)

        if total_numerosity > self.population_size:
            mean_fitness = sum(clf["fitness"] * clf["numerosity"] for clf in self.population) / total_numerosity
            vote_sum = sum(clf["numerosity"] * ((clf["fitness"] / clf["average_size"]) / mean_fitness) for clf in self.population)
            chosen_one = None
            choice = random.uniform(0, vote_sum)

            for clf in self.population:
                choice -= clf["numerosity"] * ((clf["fitness"] / clf["average_size"]) / mean_fitness)
                if choice <= 0:
                    chosen_one = clf
                    break

            if chosen_one:
                chosen_one["numerosity"] -= 1
                if chosen_one["numerosity"] == 0:
                    self.population.remove(chosen_one)
                    self.action_set = [clf for clf in self.action_set if clf != chosen_one]
                return True

        return False

    def insert_classifier(self, classifier, from_ga=False):
        if len(self.population) > self.population_size:
            self.delete_classifier()

        if from_ga:
            classifier["numerosity"] = 1
            classifier["last_time_stamp"] = self.time_stamp
            classifier["experience"] = 0
            classifier["action_set_size"] = 1
            classifier["average_size"] = len(classifier["action"])

            self.population.append(classifier)
            self.action_set.append(classifier)
        else:
            matching_classifier = self.get_classifier_with_identical_condition(classifier["condition"])
            if matching_classifier:
                matching_classifier["numerosity"] += 1
            else:
                classifier["numerosity"] = 1
                classifier["fitness"] = self.initial_error
                classifier["last_time_stamp"] = self.time_stamp
                classifier["experience"] = 0
                classifier["action_set_size"] = 0
                classifier["average_size"] = len(classifier["action"])

                self.population.append(classifier)
                self.action_set.append(classifier)

        self.subsume_classifier(classifier)

    def run_tournament(self, iteration):
            accuracy_list = []
            for i in range(100):
                j = random.randint(0, 2**self.input_size-1)
                state = format(j, f"0{self.input_size}b")

                action = self.select_action(state)

                if action is None:
                    continue

                reward = self.calculate_reward(state, action)

                accuracy = self.calculate_accuracy(action, state)
                accuracy_list.append(accuracy)

            f1_score = 0.0
            if len(accuracy_list) > 0:
                precision = sum(accuracy_list) / len(accuracy_list)
                recall = len(accuracy_list) / (2**self.input_size)
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"Iteration {iteration}, f1 score = {f1_score}")


# set the parameters for the XCS algorithm
input_size = 8
population_size = 100
crossover_rate = 0.8
mutation_rate = 0.5
theta_ga = 25
theta_as = 50
theta_exp = 50
p_sharp = 0.5
mu = 0.01
nu = 0.1
chi = 0.8
initial_error = 0.01
subsumption_threshold = 20
tournament_size = 5

# create an instance of the XCSBinaryRegex class
xcs = XCSBinaryRegex(input_size=input_size,
                     population_size=population_size,
                     crossover_rate=crossover_rate,
                     mutation_rate=mutation_rate,
                     theta_ga=theta_ga,
                     theta_as=theta_as,
                     theta_exp=theta_exp,
                     p_sharp=p_sharp,
                     mu=mu,
                     nu=nu,
                     chi=chi,
                     initial_error=initial_error,
                     subsumption_threshold=subsumption_threshold,
                     tournament_size=tournament_size)

# initialize the population of classifiers
xcs.initialize_population()

# run the XCS algorithm for a certain number of iterations
for i in range(100):
    # run a tournament and update the population of classifiers
    xcs.run_tournament(i)
    
