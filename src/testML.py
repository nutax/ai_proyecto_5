from IdentifierByTypingPatternsML import IdentifierByTypingPatternsML

input_file = "../proyecto_5/data/test.csv"

output_model_file = 'identifier.pkl'

identifier = IdentifierByTypingPatternsML.load(output_model_file)

print(identifier.test(input_file))