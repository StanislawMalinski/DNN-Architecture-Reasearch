import os
from datetime import datetime

terms = {"opt": "Optymalizator",
         "func": "Funkcja akywacji",
         "arra": "Rozkład neuronów",
         "lay": "Liczba ukrytych warstw",
         "neu": "Liczba neuronów",
         "lr": "Tępo uczenia",
         "bs": "Liczba wzorców w epoce"}

def save_results(test, win):
    to_root = os.path.abspath("Results")
    date = datetime.now().strftime("%x").replace("/", "-")
    if win:
        dir_res = f"{to_root}\\Results({date})"
    else:
        dir_res = f"{to_root}/Results({date})"
    if not os.path.isdir(dir_res):
        os.mkdir(dir_res)

    if isinstance(test, dict):
        for env in test.keys():
            if win:
                dir = f"{to_root}\\Results({date})\\{env}"
            else:
                dir = f"{to_root}/Results({date})/{env}"
            if not os.path.isdir(dir):
                os.mkdir(dir)

            for key in terms.keys():
                if key in test[env].keys():
                    df = test[env][key]
                    if win:
                        df.to_csv(f"{to_root}\\Results({date})\\{env}\\{key}.csv")
                    else:
                        df.to_csv(f"{to_root}/Results({date})/{env}/{key}.csv")
