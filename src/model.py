import tensorflow as tf
import keras
from keras.models import Model
from keras import metrics
from keras.models import Sequential
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19

from keras.layers import GlobalMaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Input

def get_encoder(input_shape):
    base_model = InceptionResNetV2(weights='imagenet', 
                                   include_top=False, 
                                   input_shape=input_shape)

    # base_model = VGG19(weights='imagenet', 
    #                                include_top=False, 
    #                                input_shape=input_shape)
    

    base_model.trainable = False

    # train_last_n = 15
    # train_after = len(base_model.layers) - train_last_n
    # for layer in base_model.layers[:train_after]:
    #     layer.trainable = False

    # for layer in base_model.layers[train_after:]:
    #     layer.trainable =  True
    
    encoder = Sequential(
        layers= [
            base_model,
            GlobalMaxPooling2D(),
            Dropout(1.0-0.8),
            Dense(128, use_bias=False),
            BatchNormalization(momentum=0.995, epsilon=0.001, scale=False),
            Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ], 
    )
    
    return encoder


def get_siamese_network(input_shape = (128, 128, 3)):
    
    # get base encoder
    encoder = get_encoder(input_shape)
    
    # Input Layers for the images
    anchor_input   = keras.layers.Input(input_shape, name="anchor_input")
    positive_input = keras.layers.Input(input_shape, name="positive_input")
    negative_input = keras.layers.Input(input_shape, name="negative_input")
    
    # Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)
    
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoded_a,
        encoded_p,
        encoded_n
    )
    
    # Creating the Model
    siamese_network = Model(
        inputs  = [anchor_input, positive_input, negative_input],
        outputs = distances,
    )
    
    return siamese_network

class DistanceLayer(keras.layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SiameseModel(Model):
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        
        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean()

    def call(self, inputs):
        return self.siamese_network(inputs)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        loss = self._compute_loss(data)
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss
    
    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]