from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import time
import pickle
import pandas as pd

def process2(models: dict, measuredDist: float, measuredDbm: float):
    answer = {}
    for model in models:
        timeStart = time.time()
        data = []
        for param in model["input"]:
            if param == "dbm": data.append(measuredDbm)
            if param == "dist": data.append(measuredDist)

            if len(data) != len(model["input"]) and param != "dbm" and param != "dist": # Pipeline model
                data.append(answer[param])

        # print(model, data)
        res = model["model"].predict(np.array(data).reshape((-1, len(data))))
        if model["name"] != "catboost" or model["name"] != "mlp": res = res[0]
        timeFinish = time.time()
        answer[f"{model['name']}_val"] = res
        answer[f"{model['name']}_freq"] = 1 / (timeFinish - timeStart)
    return answer


def pickle_model_loader(name: str):
    with open(os.path.join(PATH_TO_WEIGHTS, name), "rb") as fd:
        return pickle.load(fd)


if __name__ == "__main__":
    PATH_TO_WEIGHTS = "./weights"

    catBoost = CatBoostRegressor()
    catBoost.load_model(os.path.join(PATH_TO_WEIGHTS, "full_catboost.model"))

    testConfig = [
        {"input": ["dist", "dbm"], "name": "catboost","model": catBoost},
        {"input": ["dist"], "name": "mlp","model": pickle_model_loader("mlp_full.pkl")},
        {"input": ["dist", "dbm", "catboost_val", "mlp_val"], "name": "dtree_pipeline","model": pickle_model_loader("dt_regressor_on_ct_mlp.pkl")},
        {"input": ["dist", "catboost_val", "mlp_val"], "name": "ridge_pipeline", "model": pickle_model_loader("ridge_on_ct_mlp.pkl")},
        {"input": ["dist", "dbm", "catboost_val", "mlp_val"], "name": "svr_pipeline", "model": pickle_model_loader("svr_on_ct_mlp.pkl")},
        {"input": ["dist", "dbm", "catboost_val", "mlp_val"], "name": "rf_pipeline", "model": pickle_model_loader("rf_on_ct_mlp.pkl")},
    ]
    # print(testConfig)
    print(process2(testConfig, 2.15, -57.8))
