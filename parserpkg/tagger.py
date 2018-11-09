class Tagger():
    """
    A part-of-speech tagger based on a multi-class perceptron classifier.

    This tagger implements a simple, left-to-right tagging algorithm where the
    prediction of the tag for the next word in the sentence can be based on the
    surrounding words and the previously predicted tags. The exact features
    that this prediction is based on can be controlled with the `features()`
    method, which should return a feature vector that can be used as an input
    to the multi-class perceptron.
    """

    def __init__(self):
        self.tags = []
        self.weights = {}
        self.acc = {}
        self.count = 1
        self.words_freq = {}

    def predict(self, x):
        scores = {}

        for c in self.weights.keys():
            for f in x:
                scores[c] = scores.setdefault(c, 0.0) + self.weights[c].setdefault(f, 0.0)

        return max(scores, key = lambda c: ( scores[c], c ))

    def tag(self, words):
        """
        Tags a sentence with part-of-speech tags.
        """
        tagged_sentence = []
        previous_tags = []

        for i in range(0, len(words)):
            fv = self.features(words, i, previous_tags)

            pc = self.predict(fv)

            previous_tags.append(pc)
            tagged_sentence.append(pc)

        return tagged_sentence

    def update(self, words, gold_tags):
        """
        Updates the tagger with a single training example.
        """
        predicted_tags = []

        i = 0
        for word in words:
            self.words_freq[word] = self.words_freq.setdefault(word, 0) + 1
            fv = self.features(words, i, gold_tags)
            y = gold_tags[i]

            if y not in self.tags:
                self.tags.append(y)
                self.weights[y] = {}
                self.acc[y] = {}

            i += 1

            p = self.predict(fv)
            predicted_tags.append(p)
            if p != y:
                for f in fv:
                    self.weights[p][f] = self.weights[p].setdefault(f, 0.0) - 1
                    self.acc[p][f] = self.acc[p].setdefault(f, 0.0) - self.count

                    self.weights[y][f] = self.weights[y].setdefault(f, 0.0) + 1
                    self.acc[y][f] = self.acc[y].setdefault(f, 0.0) + self.count
        self.count += 1

        return predicted_tags

    def features(self, words, i, pred_tags):
        """
        Extracts features for the specified tagger configuration.
        """
        def has_numbers(input_string):
            return any(char.isdigit() for char in input_string)

        const = "SMOTHING?"
        t0 = pred_tags[i-1] if i > 0 else "BOS_TAG"
        t1 = pred_tags[i-2] if i > 1 else "BOS_TAG"
        t2 = pred_tags[i-3] if i > 2 else "BOS_TAG"
        w0 = words[i-1] if i > 0 else "BOS"
        w1 = words[i]
        w2 = words[i+1] if i < len(words)-1 else "EOS"
        w3 = words[i+2] if i < len(words)-2 else "EOS"
        if i > 0:
            if words[i][0].isupper():
                starts_with_cap = "STARTS WITH CAP"
            else:
                starts_with_cap = "NO CAP"
        else:
            starts_with_cap = "BOS"
        f = "FREQUENT" if self.words_freq.setdefault(words[i], 0) > 70 else "NOT FREQUENT"
        all_in_caps = words[i][0].isupper() if i > 0 else "BOS"
        long_word = "LONGER WORD" if len(words[i])> 3 else "SHORT WORD"
        ends_with_y = "ENDS WITH Y" if words[i][-1] == 'y' else "DOES NOT END WITH Y"
        one_letter_word = "ONE LETTER" if len(words[i]) == 1 else "NOT ONE LETTER"
        contains_digits = "HAS DIGITS" if has_numbers(words[i]) else "NO DIGITS"

        P = words[i][:1]
        PP = words[i][:2]
        PPP = words[i][:3]
        S = words[i][-1:]
        SS = words[i][-2:]
        SSS = words[i][-3:]

        result = [P+":"+str(0), PP+":"+str(1), PPP+":"+str(2), S, SS, SSS, one_letter_word+w1, 
        long_word, str(5)+":"+w1, ends_with_y+w1, starts_with_cap, all_in_caps, t0+w1, t1+w1+":"+str(3), const,
        w1, w1, w1, w1, t0+w0, t0+w1, t0+w2, w1+w2, w0+w1, contains_digits+w1, w2+":"+str(4), w0+w2, f]

        return result

    def finalize(self):
        """
        Averages the weight vectors.
        """
        for c in self.acc.keys():
            for f in self.acc[c]:
                self.weights[c][f] = self.weights[c].setdefault(f, 0.0) - self.acc[c].setdefault(f, 0.0) / self.count
