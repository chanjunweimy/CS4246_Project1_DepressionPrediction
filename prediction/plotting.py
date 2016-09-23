import numpy as np
import matplotlib.pyplot as plt

def plot_bar(models_rmse):
    ind = np.arange(len(models_rmse))
    width = 0.42

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rmse_train, rmse_predict = zip(*models_rmse.values())
    rects_train = ax.bar(ind, rmse_train, width, color='b')
    rects_predict = ax.bar(ind+width, rmse_predict, width, color='g')

    ax.set_ylabel('Models')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(models_rmse.keys())
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    ax.legend((rects_train[0], rects_predict[0]), ('train', 'predict'), loc=2)
    plt.title("RMSE train and predict for the different models")

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%h,
                    ha='center', va='bottom')

    autolabel(rects_train)
    autolabel(rects_predict)

    plt.show()

def plot_all_Y():
    with open("allY.txt", 'rb') as allY:
        cont = allY.readlines()

    plt.plot(range(len(cont)), cont)
    plt.show()