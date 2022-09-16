import tensorflow as tf

def create_model(loss, optimizer, metrics, classes):

    layers = [
        tf.keras.layers.Flatten(input_shape=[28,28],name='input layer'),
        tf.keras.layers.Dense(300,activation='relu',name='hidden_layer1'),
        tf.keras.layers.Dense(100,activation='relu',name='hidden_layer2'),
        tf.keras.layers.Dense(classes,activation='softmax',name='outputlayer')
    ]
    model = tf.keras.models.Sequential(layers)
    model.summary()

    #compile model
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    return model ##<<< Untrained Model
