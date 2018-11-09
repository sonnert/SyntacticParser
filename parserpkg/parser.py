from . import tagger as T
from . import classifier as C

class Parser():
    """
    A transition-based dependency parser.

    This parser implements the arc-standard algorithm for dependency parsing.
    When being presented with an input sentence, it first tags the sentence for
    parts of speech, and then uses a multi-class perceptron classifier to
    predict a sequence of moves that construct a dependency tree for the 
    input sentence. Moves are encoded as integers as follows:

    SHIFT = 0, LEFT-ARC = 1, RIGHT-ARC = 2

    At any given point in the predicted sequence, the state of the parser can
    be specified by: the index of the first word in the input sentence that
    the parser has not yet started to process; a stack holding the indices of
    those words that are currently being processed; and a partial dependency
    tree, represented as a list of indices such that `tree[i]` gives the index
    of the head (parent node) of the word at position `i`, or 0 in case the
    corresponding word has not yet been assigned a head.
    """

    def __init__(self):
        self.tagger = T.Tagger()
        self.classifier = C.Classifier()

    def parse(self, words, beam_width):
        """
        Parses a sentence.
        """
        predicted_tags = self.tagger.tag(words)

        i = 0
        stack = []
        dependency_tree = [0] * len(words)
        cfg = (i, stack, dependency_tree)

        beam = []
        beam_width = beam_width

        valid_moves = self.valid_moves(cfg[0], cfg[1], cfg[2])

        beam.append([cfg, 0, valid_moves])

        yesmilord = True

        beam_id = 0
        while yesmilord:
            temp_beam = []

            beam_id += 1

            for laser_id, laser in enumerate(beam):
                loc_cfg = laser[0]
                loc_score = laser[1]
                for loc_move in laser[2]:
                    # Predict the move
                    feature_vector = self.features(words, predicted_tags, loc_cfg[0], loc_cfg[1], loc_cfg[2])
                    predicted_move, delta_score = self.classifier.predict(feature_vector, [loc_move])

                    #tempScore = loc_score / (beam_id**2) + deltaScore
                    temp_score = delta_score

                    #print("MOVE, doing", loc_move, ", delta score: ", deltaScore, ", new score: ", tempScore)

                    # Do the move and update our index, stack and dependency_tree
                    move = self.move(loc_cfg[0], loc_cfg[1], loc_cfg[2], predicted_move)
                    temp_cfg = (move[0], move[1], move[2])

                    temp_moves = self.valid_moves(temp_cfg[0], temp_cfg[1], temp_cfg[2])

                    temp_laser = [temp_cfg, temp_score, temp_moves]

                    if len(temp_beam) < beam_width:
                        temp_beam.append(temp_laser)
                    else:
                        min_index = 0
                        min_score = temp_beam[0][1]

                        for j in range(0, len(temp_beam)):
                            if temp_beam[j][1] < min_score:
                                min_index = j
                                min_score = temp_beam[j][1]

                        if temp_laser[1] > min_score:
                            temp_beam[min_index] = temp_laser

            beam = temp_beam
            yesmilord = False
            for laser in beam:
                if len(laser[2]) > 0:
                    yesmilord = True

        max_index = 0
        max_score = beam[0][1]
        for j in range(0, len(beam)):
            if beam[j][1] > max_score:
                max_index = j
                max_score = beam[j][1]

        return (predicted_tags, beam[max_index][0][2])

    def valid_moves(self, i, stack, pred_tree):
        """
        Returns the valid moves for the specified parser configuration.
        """
        valid_moves = []

        # SHIFT = 0
        if i < len(pred_tree):
            valid_moves.append("SH")

        # LEFT-ARC = 1
        if len(stack) > 2:
            valid_moves.append("LA")

        # RIGHT-ARC = 2
        if len(stack) > 1:
            valid_moves.append("RA")

        return valid_moves

    def move(self, i, stack, pred_tree, move):
        """
        Executes a single transition.
        """

        loc_stack = []
        for j in range(0, len(stack)):
            loc_stack.append(stack[j])
        loc_tree = []
        for j in range(0, len(pred_tree)):
            loc_tree.append(pred_tree[j])

        # LEFT-ARC = 1
        if move == "LA":
            top = loc_stack.pop()
            second_top = loc_stack.pop()
            loc_tree[second_top] = top
            loc_stack.append(top)
        # RIGHT-ARC = 2
        elif move == "RA":
            top = loc_stack.pop()
            loc_tree[top] = loc_stack[-1]
        # SHIFT = 0
        else:
            loc_stack.append(i)
            i += 1

        return (i, loc_stack, loc_tree)

    def update(self, words, gold_tags, gold_tree):
        """
        Updates the move classifier with a single training example.
        """
        predicted_tags = self.tagger.update(words, gold_tags)

        i = 0
        stack = []
        dependency_tree = [ 0 ] * len(words)

        valid_moves = self.valid_moves(i, stack, dependency_tree)

        while valid_moves != []:
            # Predict the move
            feature_vector = self.features(words, predicted_tags, i, stack, dependency_tree)
            gold_move = self.gold_move(i, stack, dependency_tree, gold_tree)
            self.classifier.update(feature_vector, gold_move)

            # Do the gold move and update index, stack and dependency_tree
            move = self.move(i, stack, dependency_tree, gold_move)
            i = move[0]
            stack = move[1]
            dependency_tree = move[2]

            # Get the valid moves for the next iteration
            valid_moves = self.valid_moves(i, stack, dependency_tree)

        return (predicted_tags, dependency_tree)

    def gold_move(self, i, stack, pred_tree, gold_tree):
        """
        Returns the gold-standard move for the specified parser
        configuration.

        The gold-standard move is the first possible move from the following
        list: LEFT-ARC, RIGHT-ARC, SHIFT. LEFT-ARC is possible if the topmost
        word on the stack is the gold-standard head of the second-topmost word,
        and all words that have the second-topmost word on the stack as their
        gold-standard head have already been assigned their head in the
        predicted tree. Symmetric conditions apply to RIGHT-ARC. SHIFT is
        possible if at least one word in the input sentence still requires
        processing.
        """
        valid_moves = self.valid_moves(i, stack, pred_tree)

        if "LA" in valid_moves:
            top_most_word = stack[len(stack) - 1]
            second_top_most_word = stack[len(stack) - 2]
            second_top_most_word_head = gold_tree[second_top_most_word]

        if "RA" in valid_moves:
            top_most_word = stack[len(stack) - 1]
            top_most_word_head = gold_tree[top_most_word]
            second_top_most_word = stack[len(stack) - 2]

        # LEFT-ARC = 1
        can_do_LA = False
        if "LA" in valid_moves and second_top_most_word_head == top_most_word:
            can_do_LA = True
            for i in range(0, len(pred_tree)):
                if gold_tree[i] == second_top_most_word and pred_tree[i] == 0:
                    can_do_LA = False

        if can_do_LA:
            return "LA"

        # RIGHT-ARC = 2
        can_do_RA = False
        if "RA" in valid_moves and top_most_word_head == second_top_most_word:
            can_do_RA = True
            for i in range(0, len(pred_tree)):
                if gold_tree[i] == top_most_word and pred_tree[i] == 0:
                    can_do_RA = False

        if can_do_RA:
            return "RA"

        # SHIFT = 0
        return "SH"

    def features(self, words, tags, i, stack, parse):
        """
        Extracts features for the specified parser configuration.
        """
        def has_numbers(input_string):
            return any(char.isdigit() for char in input_string)

        features = [
            "0:" + (words[i] if i < len(words) else "<EOS>"),
            "1:" + (tags[i] if i < len(words) else "<EOS>"),
            "2:" + (words[stack[-1]] if len(stack) >= 1 else "<EMPTY>"),
            "3:" + (tags[stack[-1]] if len(stack) >= 1 else "<EMPTY>"),
            "4:" + (words[stack[-2]] if len(stack) >= 2 else "<EMPTY>"),
            "5:" + (tags[stack[-2]] if len(stack) >= 2 else "<EMPTY>")
        ]

        return features

    def finalize(self):
        """
        Averages the weight vectors.
        """
        self.classifier.finalize()
        self.tagger.finalize()
