from __future__ import print_function

header = ["taste", "color", "diameter", "label"]

# for better result add much more data to training !
training_data = [
    ['Sweet', 'Green', 3, 'Apple'],
    ['Sweet', 'Green', 4, 'Apple'],
    ['Sweet', 'Yellow', 3, 'Apple'],
    ['Sour', 'Green', 2, 'Apple'],
    ['Sour', 'Green', 3, 'Apple'],
    ['Sweet', 'Purple', 1, 'Grape'],
    ['Sweet', 'Red', 1, 'Grape'],
    ['Sweet', 'Red', 1, 'Grape'],
    ['Sour', 'Yellow', 3, 'Lemon'],
    ['Sweet', 'Yellow', 3, 'Lemon'],
    ['Sour', 'Yellow', 2, 'Lemon']
]


def unique_vals(rows, col):
    return set([row[col] for row in rows])


def class_counts(rows):
    counts = {}

    for row in rows:
        label = row[-1]

        if label not in counts:
            counts[row[-1]] = 0

        counts[row[-1]] += 1

    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


##########################Question class
class Question:

    def __init__(self, qValue, askedCol):

        self.value = qValue
        self.col = askedCol

    def match(self, row):
        test = row[self.col]
        if is_numeric(test):
            return test >= self.value  # test the < too
        else:
            return test == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is {} {} {} ?" \
            .format(
            header[self.col],
            condition,
            str(self.value)
        )


##########################END OF Question class

questionsAsked = []  # save the asked question


def partition(rows, question):
    truePart, falsePart = [], []

    for row in rows:
        if question.match(row):
            truePart.append(row)
        else:
            falsePart.append(row)

    return truePart, falsePart


# t, f = partition(training_data, Question('Apple', 2))
# print("true rows: {} \nfalse ones: {}".format(t, f))

def gini(rows):
    gini = 1
    counts = class_counts(rows)

    for row in counts:
        gini -= (counts[row] / len(rows)) ** 2
    return gini


def infoGain(currentGini, trueRows, falseRows):
    partT = len(trueRows) / (len(trueRows) + len(falseRows))

    gain = currentGini - (partT * gini(trueRows) + (1 - partT) * gini(falseRows))

    return gain


# t, f = partition(training_data, Question('Apple', 2))
# print(infoGain(gini(training_data), t, f))

def bestSplit(rows):
    bestGainSoFar = 0
    columns = len(rows[0]) - 1
    currentGini = gini(rows)
    bestQuestion = None

    for col in range(columns):

        values = unique_vals(rows, col)

        for val in values:

            question = Question(val, col)

            t, f = partition(rows, question)

            if len(t) == 0 or len(f) == 0:  # if question doesn't part the rows its not good for the classifier
                continue

            gain = infoGain(currentGini, t, f)

            if gain >= bestGainSoFar:
                bestGainSoFar, bestQuestion = gain, question

    questionsAsked.append(bestQuestion)  # don't ask this again
    return bestGainSoFar, bestQuestion


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def Tree(rows):
    best_gain, best_question = bestSplit(rows)

    if best_gain == 0:
        return Leaf(rows)

    t, f = partition(rows, best_question)

    trueTree = Tree(t)
    falseTree = Tree(f)

    return Decision_Node(best_question, trueTree, falseTree)


def printTree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    printTree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    printTree(node.false_branch, spacing + "  ")


def classify(dataRow, headOfTree):
    if isinstance(headOfTree, Leaf):
        counts = headOfTree.predictions
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"

        print("pridiction", probs)
        return

    if headOfTree.question.match(dataRow):
        classify(dataRow, headOfTree.true_branch)
    else:
        classify(dataRow, headOfTree.false_branch)


item = ['Sweet', 'Yellow', 3, 'Lemon']
print("\nItem is {} \n".format(item))
classify(item, Tree(training_data))
