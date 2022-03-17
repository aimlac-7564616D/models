import os
import pickle as pkl

root = os.path.dirname(__file__)

wind_model = None
with open(os.path.join(root, 'wind_model.pkl'), 'rb') as f:
    wind_model = pkl.load(f)