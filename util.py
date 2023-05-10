import os
from datetime import datetime

terms = {"opt": "Optymalizator",
         "func": "Funkcja akywacji",
         "arra": "Rozkład neuronów",
         "lay": "Liczba ukrytych warstw",
         "neu": "Liczba neuronów",
         "lr": "Tępo uczenia",
         "bs": "Liczba wzorców w epoce"}

def save_results(test):
    date = datetime.now().strftime("%x").replace("/", "-")

    dir_res = f"C:\\Users\\Staszek\\Documents\\Investing\\DNNReasearch\\Results\\Results({date})"
    if not os.path.isdir(dir_res):
        os.mkdir(dir_res)

    if isinstance(test, dict):
        for env in test.keys():
            dir = f"C:\\Users\\Staszek\\Documents\\Investing\\DNNReasearch\\Results\\Results({date})\\{env}"
            if not os.path.isdir(dir):
                os.mkdir(dir)

            for key in terms.keys():
                if key in test[env].keys():
                    df = test[env][key]
                    df.to_csv(
                        f"C:\\Users\\Staszek\\Documents\\Investing\\DNNReasearch\\Results\\Results({date})\\{env}\\{key}.csv")
