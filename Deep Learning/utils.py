import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt


def load_data():
    np.random.seed(1990)
    print("Loading MNIST data .....")

    # Load the MNIST dataset
    with gzip.open('Data/mnist.pkl.gz', 'r') as f:
        # u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        # train_set, valid_set, test_set = u.load()
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        train_set = [train_set[0].tolist(), [[1 if j == train_set[1][i] else 0 for j in range(10)] for i in np.arange(len(train_set[0]))]]
        valid_set = [valid_set[0].tolist(), [[1 if j == valid_set[1][i] else 0 for j in range(10)] for i in np.arange(len(valid_set[0]))]]
        test_set = [test_set[0].tolist(), [[1 if j == test_set[1][i] else 0 for j in range(10)] for i in np.arange(len(test_set[0]))]]
    print("Done.")
    return train_set, valid_set, test_set


def plot_curve(t,s,metric):
    plt.plot(t, s)
    plt.ylabel(metric) # or ERROR
    plt.xlabel('Epoch')
    plt.title('Learning Curve_'+str(metric))
    #curve_name=str(metric)+"LC.png"
    #plt.savefig(Figures/curve_name)
    plt.show()

def plot_train_val(t, st, sv, metric, error = None, xes = None):
    plt.figure(figsize=(15,10))

    if (error is not None):
        plt.subplot(1, (2 if (xes is None) else 3), 1)


    plt.plot(t, st, label='Accuracy on training set')
    plt.plot(t, sv, label='Accuracy on validation set')
    plt.ylabel(metric) # or ERROR
    plt.xlabel('Epoch')
    plt.title('Learning Curve: '+str(metric))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.grid(linestyle='--', axis="y")
    #curve_name=str(metric)+"LC.png"
    #plt.savefig(Figures/curve_name)

    if (error is not None):
        plt.subplot(1, (2 if (xes is None) else 3), 2)
        plt.plot(t, error)
        plt.ylabel('MSE') # or ERROR
        plt.xlabel('Epoch')
        plt.title('Learning Curve: Error')
        plt.grid(linestyle='--', axis="y")

    if (xes is not None):
        plt.subplot(1, 3, 3)
        plt.plot(t, error)
        plt.ylabel('X-Entropy') # or ERROR
        plt.xlabel('Epoch')
        plt.title('Learning Curve: Cross-Entropy')
        plt.grid(linestyle='--', axis="y")

    plt.show()
