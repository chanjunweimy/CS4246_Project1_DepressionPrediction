import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from collections import Counter

def plot_f1(models_f1):
    ind = np.arange(len(models_f1))
    width = 0.42

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.axhline(0.41, color='k', linestyle='solid', label="Baseline")    

    model_names, f1_depressed, f1_normal = zip(*models_f1)
    rects_normal = ax.bar(ind, f1_normal, width, color='b')
    rects_depressed = ax.bar(ind+width, f1_depressed, width, color='g')

    ax.set_ylabel('F1')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(model_names)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    ax.legend((rects_normal[0], rects_depressed[0]), ('normal', 'depresed'), loc=2)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
                    ha='center', va='bottom')

    autolabel(rects_normal)
    autolabel(rects_depressed)

    plt.show()

def plot_bar(models_rmse):
    ind = np.arange(len(models_rmse))
    width = 0.42

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.axhline(6.7418, color='k', linestyle='solid', label="Baseline")    

    model_names, rmse_train, rmse_predict = zip(*models_rmse)
    rects_train = ax.bar(ind, rmse_train, width, color='b')
    rects_predict = ax.bar(ind+width, rmse_predict, width, color='g')

    ax.set_ylabel('RMSE')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(model_names)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    ax.legend((rects_train[0], rects_predict[0]), ('train', 'predict'), loc=2)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
                    ha='center', va='bottom')

    autolabel(rects_train)
    autolabel(rects_predict)

    plt.show()

def plot_all_Y():
    with open("data/all/allY.txt", 'rb') as allY:
        cont = allY.readlines()
    cont = [ int(ea.strip()) for ea in cont]
    counter = Counter(cont)
    x = range(25)
    y = [ counter[k] for k in x]
    pylab.ylabel("Frequency")
    pylab.xlabel("Depression Severity [PHQ-8 Score]")
    plt.bar(x, y)
    plt.show()

def stats(fileName):
    with open(fileName, 'rb') as allY:
        cont = allY.readlines()
    cont = [ int(ea.strip()) for ea in cont]
    counter = Counter(cont)
    x = range(25)
    y = [ counter[k] for k in x]
    print("Std Dev:"+str(np.std(cont)))
    print("Mean:"+str(np.mean(cont)))
    print("Size:"+str(len(cont)))
