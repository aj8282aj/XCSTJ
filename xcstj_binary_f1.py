import random
import re
from xcs import XCSAlgorithm, scenarios


def generate_training_set(size):
    """
    Generates a training set of binary strings and their corresponding
    regex output.
    """
    training_set = []
    for i in range(size):
        binary_str = ''.join(random.choices(['0', '1'], k=32))
        regex_output = '1' if re.match('1[01]{3}0[01]{3}1', binary_str) else '0'
        training_set.append((binary_str, regex_output))
    return training_set


def calculate_fitness(classifier, training_set):
    """
    Calculates the fitness of a classifier on a training set by counting
    the number of examples it predicts correctly.
    """
    num_correct = 0
    for example in training_set:
        input_str, expected_output = example
        predicted_output = classifier.predict(input_str)
        if predicted_output == expected_output:
            num_correct += 1
    return num_correct / len(training_set)


# Generate a training set of 1000 examples
training_set = generate_training_set(1000)

# Initialize the XCS algorithm with default parameters
algorithm = XCSAlgorithm()

# Run the XCS algorithm on the training set for 1000 iterations
for i in range(1000):
    # Generate a new problem instance from the training set
    problem_instance = scenarios.BitStringMatch(training_set)

    # Train the XCS algorithm on the problem instance
    algorithm.run_experiment(problem_instance)

    # Calculate and print the training fitness and test f1 score every 10 iterations
    if i % 10 == 0:
        best_classifier = algorithm.population.best_classifier()
        training_fitness = calculate_fitness(best_classifier, training_set)
        test_set = generate_training_set(100)
        test_f1_score = 0
        for example in test_set:
            input_str, expected_output = example
            predicted_output = best_classifier.predict(input_str)
            if predicted_output == '1':
                predicted_regex_output = '1' if re.match('1[01]{3}0[01]{3}1', input_str) else '0'
                if predicted_regex_output == expected_output:
                    test_f1_score += 1
            else:
                if predicted_output == expected_output:
                    test_f1_score += 1
        test_f1_score /= len(test_set)
        print(f'Iteration {i}, Training fitness: {training_fitness:.2f}, Test f1 score: {test_f1_score:.2f}')
