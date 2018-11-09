def conllu(fp):
    block = []
    for row in fp:
        if row[0] == '#':
            continue
        if row not in ['\n', '\r\n']:
            inner = []
            s = row.split()

            for w in s:
                inner.append(w)

            block.append(inner)
        
        if row in ['\n', '\r\n'] and len(block) > 0:
            yield block
            block = []

def trees(fp):
    """
    Reads trees from an input source.
    """

    for sentence in conllu(fp):
        result = ([ "<ROOT>" ], [ "<ROOT>" ], [ 0 ])
        for word in sentence:
            result[0].append(word[1])
            result[1].append(word[3])
            result[2].append(int(word[6]))
        yield result
