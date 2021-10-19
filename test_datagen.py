from utils.dataloader import  DataLoader
import json

#load the params-tiles.json options
with open('params-tiles.json') as param_file:
    params_tiles = json.load(param_file)


dl = DataLoader