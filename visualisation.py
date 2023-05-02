import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from util import terms


def visualise(data: pd.DataFrame, title="DNN reaserch"):
    data.plot()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

def visualise_folder(path):
    if not os.path.isdir(path):
        Exception("Podany folder nie istnieje.")

    files = [path + "\\" + f for f in os.listdir(path)]

    for file in files:
        if os.path.isdir(file):
            files += [file + "\\" + f for f in os.listdir(file)]
        else:
            env = re.findall(r'\\([\w\s]*)\\(\w+).csv', file)[0]
            df = pd.read_csv(file).drop(columns=['Unnamed: 0'])
            visualise(df, str(env[0]) + " - " + terms[str(env[1])])


if __name__ == "__main__":
    dir = "/Results/Results(05-02-23(1))"
    visualise_folder(dir)
