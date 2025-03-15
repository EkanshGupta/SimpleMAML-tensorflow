from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

class EEGNet(tf.keras.Model):
    def __init__(self, nb_classes=2,Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25):
        super(EEGNet, self).__init__()
            
        ##################################################################
        self.block1       = Conv2D(F1, (1, kernLength), padding = 'same',input_shape = (Chans, Samples, 1),use_bias = False)
        self.block2       = BatchNormalization()
        self.block3       = DepthwiseConv2D((Chans, 1), use_bias = False, depth_multiplier = D,depthwise_constraint = max_norm(1.))
        self.block4       = BatchNormalization()
        self.block5       = Activation('elu')
        self.block6       = AveragePooling2D((1, 4))
        self.block7       = Dropout(dropoutRate)
        self.block8       = SeparableConv2D(F2, (1, 16),use_bias = False, padding = 'same')
        self.block9       = BatchNormalization()
        self.block10       = Activation('elu')
        self.block11       = AveragePooling2D((1, 8))
        self.block12       = Dropout(dropoutRate)
        self.flatten      = Flatten(name = 'flatten')
        self.dense        = Dense(nb_classes, name = 'dense')
        self.softmax      = Activation('softmax', name = 'softmax')

        # Add a list of layer blocks to make iteration easier
        self.layer_blocks = [
            self.block1, self.block2, self.block3, self.block4,
            self.block5, self.block6, self.block7, self.block8,
            self.block9, self.block10, self.block11, self.block12,
            self.flatten, self.dense, self.softmax
        ]
        
    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

    # Method to get trainable weights in order
    def get_ordered_weights(self):
        ordered_weights = []
        for layer in self.layer_blocks:
            if layer.trainable_weights:
                ordered_weights.extend(layer.trainable_weights)
        return ordered_weights



def fastWeights_EEGNet(model, custom_weights, inputs):
    """
    Apply custom weights to the EEGNet model in a forward pass.
    
    Args:
        model: An instance of EEGNet
        custom_weights: List of custom weight tensors to use instead of model's weights
        inputs: Input tensor to the model
        
    Returns:
        Output tensor after forward pass with custom weights
    """
    # Get original weights for reference
    original_weights = model.get_ordered_weights()
    
    # Check if the number of weights match
    if len(custom_weights) != len(original_weights):
        raise ValueError(f"Number of custom weights ({len(custom_weights)}) "
                         f"doesn't match model weights ({len(original_weights)})")
    
    # Create a mapping of layer to its custom weights
    weight_mapping = {}
    weight_idx = 0
    
    for layer in model.layer_blocks:
        if layer.trainable_weights:
            num_weights = len(layer.trainable_weights)
            weight_mapping[layer] = custom_weights[weight_idx:weight_idx + num_weights]
            weight_idx += num_weights
    
    # Forward pass with custom weights
    x = inputs

    # print(x.shape)
    
    # Block 1: Conv2D (no bias)
    conv_weights = weight_mapping[model.block1][0]
    x = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # print(x.shape)
    
    # Block 2: BatchNormalization
    gamma, beta, moving_mean, moving_var = model.block2.weights
    if model.block2 in weight_mapping:
        gamma = weight_mapping[model.block2][0]
        beta = weight_mapping[model.block2][1]
    x = tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma, 1e-3)
    # print(x.shape)
    
    # Block 3: DepthwiseConv2D (no bias)
    dw_weights = weight_mapping[model.block3][0] if model.block3 in weight_mapping else model.block3.weights[0]
    # print(f"The shape of depthwise weights: {dw_weights.shape}")
    x = tf.nn.depthwise_conv2d(x, dw_weights, strides=[1, 1, 1, 1], padding='VALID')
    # print(x.shape)
    
    # Block 4: BatchNormalization
    gamma, beta, moving_mean, moving_var = model.block4.weights
    if model.block4 in weight_mapping:
        gamma = weight_mapping[model.block4][0]
        beta = weight_mapping[model.block4][1]
    x = tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma, 1e-3)
    # print(x.shape)
    
    # Block 5: Activation (ELU)
    x = tf.nn.elu(x)
    # print(x.shape)
    
    # Block 6: AveragePooling2D
    x = tf.nn.avg_pool2d(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='VALID')
    # print(x.shape)
    
    # Block 7: Dropout (not applied during inference)
    # Skip dropout during fast weights application
    
    # Block 8: SeparableConv2D
    # For SeparableConv2D we need depthwise and pointwise parts
    if model.block8 in weight_mapping:
        depthwise_kernel = weight_mapping[model.block8][0]
        pointwise_kernel = weight_mapping[model.block8][1]
    else:
        depthwise_kernel = model.block8.depthwise_kernel
        pointwise_kernel = model.block8.pointwise_kernel
    
    # Apply depthwise conv
    x = tf.nn.depthwise_conv2d(x, depthwise_kernel, strides=[1, 1, 1, 1], padding='SAME')
    # Apply pointwise conv
    x = tf.nn.conv2d(x, pointwise_kernel, strides=[1, 1, 1, 1], padding='SAME')
    # print(x.shape)
    
    # Block 9: BatchNormalization
    gamma, beta, moving_mean, moving_var = model.block9.weights
    if model.block9 in weight_mapping:
        gamma = weight_mapping[model.block9][0]
        beta = weight_mapping[model.block9][1]
    x = tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma, 1e-3)
    # print(x.shape)
    
    # Block 10: Activation (ELU)
    x = tf.nn.elu(x)
    # print(x.shape)
    
    # Block 11: AveragePooling2D
    x = tf.nn.avg_pool2d(x, ksize=[1, 1, 8, 1], strides=[1, 1, 8, 1], padding='VALID')
    # print(x.shape)
    
    # Block 12: Dropout (not applied during inference)
    # Skip dropout during fast weights application
    
    # Flatten
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, [batch_size, -1])
    # print(x.shape)
    
    # Dense
    if model.dense in weight_mapping:
        kernel = weight_mapping[model.dense][0]
        bias = weight_mapping[model.dense][1]
    else:
        kernel = model.dense.kernel
        bias = model.dense.bias
    x = tf.matmul(x, kernel) + bias
    # print(x.shape)
    
    # Softmax activation
    x = tf.nn.softmax(x)
    
    return x
