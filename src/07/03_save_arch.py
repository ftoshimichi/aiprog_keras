json_string = model.to_json()
f = open("my_model_arch.json", "w")
f.write(json_string)
f.close()
