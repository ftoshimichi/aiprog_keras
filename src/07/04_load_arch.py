from keras.models import model_from_json

f = open("my_model_arch.json", "r")
json_string = f.read()
f.close()

model2 = model_from_json(json_string)
print(len(model2.layers))
