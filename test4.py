import re
import numpy as np
import pygad
from sklearn import svm
from sklearn.metrics import f1_score

# Define the regular expression pattern
pattern = r'a+b+'

# Define the LCS
class LCS:
    def __init__(self, clf):
        self.clf = clf
        
    def fit(self, X, y):
        self.clf.fit(X, y)
        
    def predict(self, X):
        return self.clf.predict(X)
    
    def score(self, X, y_true):
        y_pred = self.clf.predict(X)
        return f1_score(y_true, y_pred,average='macro')
    
    def generate_rule(self, features):
        regex = ""
        for feature in features:
            if feature == 0:
                regex += "."
            elif feature == 1:
                regex += "[a-zA-Z]"
            elif feature == 2:
                regex += "[^a-zA-Z]"
        return regex

# Define the fitness function for PyGAD
def fitness_function(solution, solution_idx):
    # Generate the rule from the solution
    rule = lcs.generate_rule(solution)
    # Compile the regular expression pattern
    regex = re.compile(rule)
    # Generate random training data
    X = np.random.randint(0, 26, size=(250, 32   )).astype(np.chararray)
    y =  np.random.randint(0, 26,250).astype(np.uint8)
    X = np.reshape(X, (len(X), -1))
    # Train the LCS and calculate F1 score
    lcs.fit(X, y)
    f1 = lcs.score(X, y)
    print("Solution {0} - F1 score: {1:.4f}".format(solution_idx, f1))
    # Return the F1 score as the fitness value
    return f1

# Define the PyGAD configuration
num_generations = 100
num_parents_mating = 10
sol_per_pop = 20
num_genes = 10
gene_space = [0, 1, 2]
parent_selection_type = "rank"
mutation_type = "random"
mutation_percent_genes = 10

# Define the SVM classifier
clf = svm.SVC(kernel='linear')

# Create an instance of the LCS class
lcs = LCS(clf)

# Create an instance of the PyGAD library
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       fitness_func=fitness_function)

# Run the PyGAD optimization process
ga_instance.run()

# Get the best solution and generate the final regular expression pattern
best_solution = ga_instance.best_solution()[0]
final_rule = lcs.generate_rule(best_solution)
print("Final regular expression pattern: ", final_rule)
# Evaluate the LCS using F1 score


