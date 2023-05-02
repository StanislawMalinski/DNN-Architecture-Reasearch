import pandas as pd
import matplotlib.pyplot as plt
from util import terms


def visualise(data: pd.DataFrame, title="DNN reaserch"):
    data.plot()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.show()

def visualise_dic(data :dict):
    for key in data.keys():
        visualise(data[key], terms[key])

if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\Staszek\\Documents\\Investing\\DNNReasearch\\Results\\Results(04-24-23)\\arra.txt").drop(columns=['Unnamed: 0'])
    visualise(df)
