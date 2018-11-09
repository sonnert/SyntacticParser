import sys, os

import parserpkg.classifier as C
import parserpkg.tagger as T
import parserpkg.parser as P
import parserpkg.model_storer as S
from formattingpkg.library import *

n_examples = None    # Set to None to train on all examples

if (len(sys.argv) == 4 and sys.argv[3] == "yes") and os.path.isfile(sys.argv[1] + ".trained.json"):
    parser = S.read_model(sys.argv[1] + ".trained.json")
else:
    parser = P.Parser()
    with open(sys.argv[1]) as fp:
        for i, (words, gold_tags, gold_tree) in enumerate(trees(fp)):
            parser.update(words, gold_tags, gold_tree)
            print("\rUpdated with sentence #{}".format(i), end="")
            if n_examples and i >= n_examples:
                break
        print("")
    parser.finalize()
    S.store_model(parser, sys.argv[1] + ".trained.json")

with open(sys.argv[2]) as fp:
    for i, line in enumerate(fp):
        if line[0] == '#' or line in ['\n', '\r\n']:
            continue
        
        words = line.split()
        pred_tags, pred_tree = parser.parse(words)

        root = None
        right = []
        left = []

        for j in range(0, len(pred_tree)):
            if pred_tree[j] == 0 and pred_tags[j] == "VERB":
                root = j

        for j in range(0, len(pred_tree)):
            if pred_tree[j] == root and pred_tags[j] == "NOUN":
                if j < root:
                    left.append(words[j])
                else:
                    right.append(words[j])

        print(words)
        print(pred_tags)
        print(pred_tree)
        if root is not None:
            print("LEFT: ", left, "ROOT: ", words[root], ", RIGHT: ", right)
        else:
            print("COULD NOT FIND VERB ROOT!")
        print("")

print("")
