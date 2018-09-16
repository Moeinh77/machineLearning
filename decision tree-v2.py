from __future__ import print_function

training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

header = ["color", "diameter", "label"]


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


print(gini(training_data))


def infoGain(currentGini, trueRows, falseRows):
    partT = len(trueRows) / (len(trueRows) + len(falseRows))

    gain = currentGini - (partT * gini(trueRows) + (1 - partT) * gini(falseRows))

    return gain


t, f = partition(training_data, Question('Apple', 2))
print(infoGain(gini(training_data), t, f))
