from titli.ids import LOF
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset

batch_size = 32
model = LOF()

data = pd.read_csv("./utils/weekday_20k.csv")

feature, label = data.iloc[:, :-1].values, data.iloc[:, -1].values
dataset = TensorDataset(torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train(dataloader)
model.save_model(f"{model.__class__.__name__}_model.pkl")
model.load_model(f"{model.__class__.__name__}_model.pkl")
y_test, y_pred = model.infer(dataloader)

results = model.evaluate(y_test, y_pred)
model.plot(results)

from pprint import pprint
pprint(results)
