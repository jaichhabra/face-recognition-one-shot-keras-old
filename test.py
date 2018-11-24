from keras.models import load_model
from load_data import data

path = input("Path: ")

model = load_model("models/"+path+"/model.h5")

left,right,out = data(50)

run = model.predict([left,right])

print(run)

print(out)
