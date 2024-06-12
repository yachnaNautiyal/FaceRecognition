import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.layers import (Conv2D, ZeroPadding2D, Activation, Input, concatenate,
                                     Dense, Lambda, Flatten, BatchNormalization, MaxPooling2D,
                                     AveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Concatenate


print("started")
# Custom Local Response Normalization (LRN) layer
def lrn(x):
    return tf.nn.local_response_normalization(x, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

# Custom triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

# Function to build the Inception-based FaceNet model
def build_facenet(input_shape):
    myInput = Input(shape=input_shape)

    # Layer 1
    x = ZeroPadding2D(padding=(3, 3))(myInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Layer 2
    x = Conv2D(64, (1, 1), strides=(1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), strides=(1, 1), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Layer 3a
    inception_3a_3x3 = Conv2D(128, (1, 1), name='inception_3a_3x3_reduce')(x)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3a_3x3_reduce')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
    inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
    inception_3a_3x3 = Conv2D(192, (3, 3), name='inception_3a_3x3')(inception_3a_3x3)
    inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3a_3x3')(inception_3a_3x3)
    inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

    inception_3a_5x5 = Conv2D(32, (1, 1), name='inception_3a_5x5_reduce')(x)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3a_5x5_reduce')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
    inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
    inception_3a_5x5 = Conv2D(96, (5, 5), name='inception_3a_5x5')(inception_3a_5x5)
    inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3a_5x5')(inception_3a_5x5)
    inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

    inception_3a_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_proj')(inception_3a_pool)
    inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='bn3a_pool_proj')(inception_3a_pool)
    inception_3a_pool = Activation('relu')(inception_3a_pool)

    inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1')(x)
    inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3a_1x1')(inception_3a_1x1)
    inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

    # Ensure all branches have the same spatial dimensions
    inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=-1)

    # Layer 3b
    inception_3b_3x3 = Conv2D(128, (1, 1), name='inception_3b_3x3_reduce')(inception_3a)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3b_3x3_reduce')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
    inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
    inception_3b_3x3 = Conv2D(192, (3, 3), name='inception_3b_3x3')(inception_3b_3x3)
    inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3b_3x3')(inception_3b_3x3)
    inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

    inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_reduce')(inception_3a)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3b_5x5_reduce')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
    inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
    inception_3b_5x5 = Conv2D(96, (5, 5), name='inception_3b_5x5')(inception_3b_5x5)
    inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3b_5x5')(inception_3b_5x5)
    inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inception_3a)
    inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_proj')(inception_3b_pool)
    inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='bn3b_pool_proj')(inception_3b_pool)
    inception_3b_pool = Activation('relu')(inception_3b_pool)

    inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1')(inception_3a)
    inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='bn3b_1x1')(inception_3b_1x1)
    inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

    # Ensure all branches have the same spatial dimensions
    inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=-1)

    x = ZeroPadding2D(padding=(1, 1))(inception_3b)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Fully connected layer
    x = Flatten()(x)
    x = Dense(128, name='dense_layer')(x)
    x = BatchNormalization(name='bn_dense_layer')(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1))(x)

    model = Model(inputs=[myInput], outputs=x)
    return model

# Build the model
input_shape = (160, 160, 3)  # Adjust the input shape if necessary
facenet_model = build_facenet(input_shape)
facenet_model.compile(optimizer='adam', loss=triplet_loss)

# Save the model
model_path = 'models/facenet.pb'
tf.keras.models.save_model(facenet_model, model_path)
print(f"Model saved to {model_path}")