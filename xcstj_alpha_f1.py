import re
import random
from collections import defaultdict
import numpy as np

class Classifier:
    def __init__(self, condition, action, prediction, error, fitness, num, timestamp, exp, action_set_size, average_size, utility):
        self.condition = condition
        self.action = action
        self.prediction = prediction
        self.error = error
        self.fitness = fitness
        self.num = num
        self.timestamp = timestamp
        self.exp = exp
        self.action_set_size = action_set_size
        self.average_size = average_size
        self.utility = utility

new_classifier = Classifier(condition="^.{32}$", action=1, prediction=0.5, error=0.2, fitness=0.6, num=10, timestamp=123456, exp=5, action_set_size=20, average_size=10, utility=0.7)

class XCSAlphabeticRegex:
    def __init__(self, population_size: int = 1000, learning_rate: float = 0.1, discount_factor: float = 0.1,
                 threshold: int = 20, ga_threshold: int = 100, crossover_probability: float = 0.8,
                 mutation_probability: float = 0.04, deletion_probability: float = 0.01, wildcard_probability: float = 0.05,
                 error_threshold: float = 0.01, input_size: int = 32, target_string: str = None):
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.ga_threshold = ga_threshold
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.deletion_probability = deletion_probability
        self.wildcard_probability = wildcard_probability
        self.error_threshold = error_threshold
        self.input_size = input_size
        self.target_string = target_string
        self.time_stamp = 0
        self.classifiers = []
        self.action_set = []
        self.match_set = []
        self.accuracy_list = []
        self.f1_score = 0

    def calculate_reward(self, state: str, action: str) -> float:
        if action == self.target_string:
            return 1000.0
        else:
            return -1.0

    def select_action(self, state: str) -> str:
        actions = [c.action for c in self.action_set if c.is_match(state)]
        if actions:
            return random.choice(actions)
        return None

    def calculate_accuracy(self, action: str, state: str) -> float:
        return sum([action[i] == state[i] for i in range(self.input_size)]) / self.input_size

    def insert_classifier(self, classifier):
        self.classifiers.append(classifier)

    def delete_classifier(self, classifier):
        self.classifiers.remove(classifier)

    def subsume_classifier(self, classifier):
        for cl in self.classifiers:
            if cl.is_more_general(classifier):
                cl.numerosity += 1
                return
        self.insert_classifier(classifier)

    def update_action_set(self):
        self.action_set = [cl for cl in self.match_set if cl.action is not None]

    def update_f1_score(self):
        tp = 0
        fp = 0
        fn = 0
        for i in range(2**self.input_size):
            state = format(i, f"0{self.input_size}b")
            action = self.select_action(state)
            if action is None:
                continue
            accuracy = self.calculate_accuracy(action, state)
            if accuracy == 1:
                if re.match("^" + self.get_regex() + "$", state):
                    tp += 1
                else:
                    fp += 1
            else:
                if re.match("^" + self.get_regex() + "$", state):
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        self.f1_score = 2 * (precision * recall) / (precision + recall)

    def get_fitness_sum(self):
        return sum([cl.fitness for cl in self.classifiers])

    def get_random_bits(self):
        return format(random.randint(0, 2**self.input_size-1), f"0{self.input_size}b")
    def get_regex(self):
        regex = ""
        for i in range(self.num_bits):
            classifier = self.select_action(format(i, f"0{self.input_size}b"))
            if classifier is None:
                regex += "[a-z]"
            else:
                regex += classifier.get_prediction()

        return regex
    def select_action(self, state):
        matching_classifiers = [cl for cl in self.action_set if cl.matches(state)]
        if matching_classifiers:
            action_counts = defaultdict(int)
            for cl in matching_classifiers:
                action_counts[cl.action] += cl.numerosity
            return max(action_counts, key=action_counts.get)
        else:
            return None
    def run_tournament(self, iteration):
        accuracy_list = []
        for i in range(100):
            state = self.get_random_bits()
            action = self.select_action(state)

            if action is None:
                continue

            reward = self.calculate_reward(state, action)

            accuracy = self.calculate_accuracy(action, state)
            accuracy_list.append(accuracy)

        f1_score = self.update_f1_score()
        print(f"Iteration: {iteration} | F1 Score: {f1_score:.2f}")
    def train(self, iterations):
        for i in range(iterations):
            self.run_tournament(i)
            if i % self.learning_rate == 0:
                self.delete_weak_classifiers()
                self.subsume_classifiers()
                self.generate_covering_classifier()
                self.update_f1_score()
    def generate_covering_classifier(self, state):
        pattern = ''
        for i in range(self.input_size):
            if random.random() < self.p_sharp:
                pattern += '\w'
            else:
                pattern += state[i]

        classifier = Classifier(pattern, '1' * self.input_size, '0' * self.input_size)
        self.insert_classifier(classifier)

        return classifier
    def subsume_classifiers(self, classifiers):
            cl = []
            for c in classifiers:
                if c.is_subsumable():
                    for sc in cl:
                        if c.subsumes(sc):
                            sc.numerosity += 1
                            break
                    else:
                        cl.append(c)
                else:
                    cl.append(c)

            while sum([c.numerosity for c in cl]) > self.max_population_size:
                least_fit_index = self.get_least_fit_classifier_index(cl)
                c = cl[least_fit_index]
                if c.numerosity > 1:
                    c.numerosity -= 1
                else:
                    cl.pop(least_fit_index)

            return cl
    def delete_weak_classifiers(self):
        average_fitness = np.mean([c.fitness for c in self.classifiers])
        self.classifiers = [c for c in self.classifiers if c.fitness >= self.theta_del * average_fitness]
xcs = XCSAlphabeticRegex(32)

# train the classifier for 100 iterations
xcs.train(100)

# get the learned regular expression
regex = xcs.get_regex()
print("Learned regular expression:", regex)

# test the regular expression with a random string
test_string = xcs.get_random_bits()
print("Test string:", test_string)
match = re.match("^" + regex + "$", test_string)
if match:
    print("String matched regular expression!")
else:
    print("String did not match regular expression.")