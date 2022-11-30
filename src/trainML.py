from IdentifierByTypingPatternsML import IdentifierByTypingPatternsML

input_file = "../proyecto_5/data/ai_project_public_typing_info.csv"

output_model_file = 'identifier.pkl'

identifier = IdentifierByTypingPatternsML()

score = identifier.train(input_file, test_size=0.3)

identifier.store(output_model_file)

print(score)
