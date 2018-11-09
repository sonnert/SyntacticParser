import sys, os

import parserpkg.classifier as C
import parserpkg.tagger as T
import parserpkg.parser as P
import parserpkg.model_storer as S
from formattingpkg.library import *

n_examples = None    # Set to None to train on all examples

#beam_width = int(sys.argv[3])

if (len(sys.argv) == 5 and sys.argv[4] == "yes") and os.path.isfile(sys.argv[1] + ".trained.json"):
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

acc_k = acc_n = 0
uas_k = uas_n = 0
with open(sys.argv[2]) as fp:
    for i, (words, gold_tags, gold_tree) in enumerate(trees(fp)):
        pred_tags, pred_tree = parser.parse(words, beam_width)
        acc_k += sum(int(g == p) for g, p in zip(gold_tags, pred_tags)) - 1
        acc_n += len(words) - 1
        uas_k += sum(int(g == p) for g, p in zip(gold_tree, pred_tree)) - 1
        uas_n += len(words) - 1
        print("\rParsing sentence #{}".format(i), end="")
    print("")
print("Tagging accuracy: {:.2%}".format(acc_k / acc_n))
print("Unlabelled attachment score: {:.2%}".format(uas_k / uas_n))
