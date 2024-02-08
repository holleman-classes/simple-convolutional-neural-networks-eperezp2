### Add lines to import modules as needed
import tensorflow as tf
from tensorflow.keras import models, layers
## 

def build_model1():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model


def build_model2():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    return model


def build_model3():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_1_output = layers.add([x, inputs])

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation='relu')(block_2_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def build_model50k():
    model = models.Sequential([
        layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model


# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set

  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  
  ### Repeat for model 3 and your best sub-50k params model
  
  
