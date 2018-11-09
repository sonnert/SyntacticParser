import json
from . import parser as P

def store_model(model, filename):
    model_struct = {'classifier':{'weights':model.classifier.weights, 'classes':model.classifier.classes},
                   'tagger':{'weights':model.tagger.weights, 'tags':model.tagger.tags}}
    model_string = json.dumps(model_struct)
    file = open(filename, 'w')
    file.write(model_string)
    file.close()

    new_model = read_model(filename)
    if new_model.classifier.weights != model.classifier.weights:
        print("Classifier weights differ!")
        print(model.classifier.weights)
        print(new_model.classifier.weights)
    if new_model.classifier.classes != model.classifier.classes:
        print("Classifier classes differ!")
    if new_model.tagger.weights != model.tagger.weights:
        print("Tagger weights differ!")
    if new_model.tagger.tags != model.tagger.tags:
        print("Tagger tags differ!")

def read_model(file_name):
    file = open(file_name, 'r')
    model_string = file.read()
    model_struct = json.loads(model_string)
    model = P.Parser()
    model.classifier.weights = model_struct['classifier']['weights']
    model.classifier.classes = model_struct['classifier']['classes']
    model.tagger.weights = model_struct['tagger']['weights']
    model.tagger.tags = model_struct['tagger']['tags']

    return model
