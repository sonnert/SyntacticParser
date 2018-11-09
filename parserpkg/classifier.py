class Classifier():
    """
    A multi-class perceptron classifier.

    A multi-class perceptron consists of a number of cells, one cell for each
    class. When a cell is presented with an input, it gets activated, and the
    cell with the highest activation predicts the class for the input. An input
    is represented by a feature vector, and the activation is computed by
    taking the dot product (weighted sum) of this feature vector and the
    cell-specific weight vector.

    This implementation of a multi-class perceptron assumes that both classes
    and features can be used as dictionary keys. Feature vectors are
    represented as lists of features.
    
    """
    def __init__(self):
        self.classes = []
        self.weights = {}

        self.acc = {}
        self.count = 1

    def predict(self, x, candidates=None):
        """
        Predicts the class for a feature vector.

        This computes the activations for the classes of this perceptron for
        the feature vector `x` and returns the class with the highest
        activation.
        """
        scores = {}

        occurrences = {}
        for word in x:
            occurrences[word] = occurrences.setdefault(word, 0) + 1

        keys = self.weights.keys()
        if candidates is not None:
            keys = candidates

        for c in keys:
            for f in occurrences:
                scores[c] = scores.setdefault(c, 0.0) + self.weights[c].setdefault(f, 0.0) * occurrences[f]

        key = max(scores, key = lambda c: ( scores[c], c ))

        return key, scores[key]

    def update(self, x, y):
        """
        Updates the weight vectors with a single training example.
        """
        if y not in self.classes:
            self.classes.append(y)
            self.weights[y] = {}
            self.acc[y] = {}

        p, s = self.predict(x)
        if p != y:
            for f in x:
                self.weights[p][f] = self.weights[p].setdefault(f, 0.0) - 1
                self.acc[p][f] = self.acc[p].setdefault(f, 0.0) - self.count

                self.weights[y][f] = self.weights[y].setdefault(f, 0.0) + 1
                self.acc[y][f] = self.acc[y].setdefault(f, 0.0) + self.count
        self.count += 1

        return p

    def finalize(self):
        """
        Averages the weight vectors.
        """
        for c in self.acc.keys():
            for f in self.acc[c]:
                self.weights[c][f] = self.weights[c].setdefault(f, 0.0) - self.acc[c].setdefault(f, 0.0) / self.count
