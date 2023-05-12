import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from util import terms

BACK_SLASH = False

def visualise(data: pd.DataFrame, title="DNN reaserch"):
    data.plot()
    plt.xlabel('Epoch')
    plt.ylabel('RMS')
    plt.title(title)
    plt.show()

def visualise_folder(path, worst=-1, best=-1):
    if not os.path.isdir(path):
        Exception("Podany folder nie istnieje.")
    if BACK_SLASH:
        files = [path + "\\" + f for f in os.listdir(path)]
        for file in files:
            if os.path.isdir(file):
                files += [file + "\\" + f for f in os.listdir(file)]
            else:
                env = re.findall(r'\\([\w\s]*)\\(\w+).csv', file)[0]
                df = pd.read_csv(file).drop(columns=['Unnamed: 0'])
                __show_figure(df, env, best, worst)
    else:
        files = [path + "/" + f for f in os.listdir(path)]
        for file in files:
            if os.path.isdir(file):
                files += [file + "/" + f for f in os.listdir(file)]
            else:
                env = re.findall(r'/([\w\s]*)/(\w+).csv', file)[0]
                df = pd.read_csv(file).drop(columns=['Unnamed: 0'])
                __show_figure(df, env, best, worst)


def __show_figure(df, names, best, worst):
    s = df.sum().sort_values()
    label = []
    if best > 0:
        bs = s[:best]
        label += [name for name in bs.index]
        if worst > 0:
            ws = s[-worst:]
            label += [name for name in ws.index]
    else:
        if worst > 0:
            ws = s[-worst:]
            label += [name for name in ws.index]
        else:
            label = s.index

    visualise(df[label], str(names[0]) + " - " + terms[str(names[1])])

if __name__ == "__main__":
    dir = "/Users/stanislaw/PycharmProjects/DNNreasearch/Results/Results(05-03-23)"
    visualise_folder(dir,worst=1,best=2 )
