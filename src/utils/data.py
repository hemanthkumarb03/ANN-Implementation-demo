import tensorflow as tf


def get_data(val):

    #loading the data
    mnist = tf.keras.datasets.mnist
    (x_train_full,y_train),(x_test,y_test)=mnist.load_data()

    #Split the dataset
    x_val, y_val = x_train_full[:val] , y_train[:val]
    x_train, y_train = x_train_full[val:], y_train[val:]

    #scale the dataset
    x_train=x_train/255.
    x_val=x_val/255.
    x_test= x_test/255.

    return (x_train,y_train),(x_val,y_val),(x_test,y_test)