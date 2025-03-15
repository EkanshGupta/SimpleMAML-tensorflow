import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import metrics
import globals
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import clone_model
from copy import deepcopy, copy
import gc
import tracemalloc
from random import shuffle
import matplotlib.pyplot as plt

# Function to initialize and compile EEGNet model
def initialize_eegnet():
    """Create, compile, and return the EEGNet model."""
    model = EEGNet(nb_classes=2, Chans=globals.chans, Samples=globals.samples,
                   dropoutRate=globals.model_dropout, kernLength=globals.model_kern,
                   F1=globals.model_f1, D=globals.model_d, F2=globals.model_f2,
                   dropoutType='Dropout')
    return model

class MAML:
    def __init__(self, input_shape=(64, 128, 1), num_classes=2, inner_lr=0.001, outer_lr=0.001, 
                 inner_steps=5, outer_steps=100, fine_tune_steps = 10, num_users=5, support_size=20, query_size=20):
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.fine_tune_steps = fine_tune_steps
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.support_size = support_size
        self.query_size = query_size
        self.num_users = num_users
        self.model = self.create_eegnet(num_classes)
        self.meta_optimizer = tf.keras.optimizers.Adam(outer_lr)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def create_eegnet(self,num_classes):
        eegnet = EEGNet(nb_classes=num_classes,Chans=globals.chans, Samples=globals.samples, dropoutRate = globals.model_dropout, 
                kernLength = globals.model_kern, F1 = globals.model_f1, D = globals.model_d, F2 = globals.model_f2)  
        eegnet.build((None, globals.chans, globals.samples,1))
        return eegnet
                
    def updateMAML(self, test_users):
        def taskLoss(batch):
            support_x,support_y,query_x,query_y = batch
            with tf.GradientTape() as taskTape:
                loss = self.loss_fn(support_y, self.model(support_x))
            
            grads = taskTape.gradient(loss, self.model.trainable_weights)
            weights = [w - self.inner_lr * g for g, w in zip(grads, self.model.trainable_weights)]
            return self.loss_fn(query_y, fastWeights_EEGNet(self.model, weights, query_x))
    
        batch=[tf.Variable([1.0]),tf.constant([1.0]),tf.Variable([1.0]),tf.constant([1.0])]
        total_train_loss=0
        test_acc_list = [[] for i in range(len(test_users))]
        for idx,user in enumerate(test_users):
            support_x, support_y = globals.train_data[user]["support"]
            query_x, query_y = globals.train_data[user]["query"]
            test_x, test_y = globals.train_data[user]["test"]
            support_x = support_x.reshape(support_x.shape[0], globals.chans, globals.samples, 1)
            query_x = query_x.reshape(query_x.shape[0], globals.chans, globals.samples, 1)
            test_x = test_x.reshape(test_x.shape[0], globals.chans, globals.samples, 1)
            support_y      = np_utils.to_categorical(support_y)
            query_y      = np_utils.to_categorical(query_y)
            test_y      = np_utils.to_categorical(test_y)
            with tf.GradientTape() as tape:	
                # loss = tf.map_fn(taskLoss, elems=(support_x,support_y,query_x,query_y),fn_output_signature=tf.float32)
                # loss = tf.reduce_sum(batchLoss)
                loss = taskLoss([support_x,support_y,query_x,query_y])
                total_train_loss+=loss
            meta_gradients = tape.gradient(loss, self.model.trainable_variables)
            self.meta_optimizer.apply_gradients(zip(meta_gradients, self.model.trainable_variables))
        avg_train_loss = total_train_loss / (len(test_users))
        print(f"Train Loss: {avg_train_loss:.4f}")
        return avg_train_loss

    def train(self, train_user, test_users, pretraining_epochs=100, pretrain_lr=0.001, pretrain=False):
        train_loss_list=[]
        self.model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=pretrain_lr),
          metrics=[metrics.SpecificityAtSensitivity(0.5, num_thresholds=50)])
        print(self.model.summary)

        for step in range(self.outer_steps):
            train_loss_list.append(self.updateMAML(test_users))
            if step % 1 == 0:
                print(f"Step {step}/{self.outer_steps} completed")
        return self.model

    def test_model(self, test_users):
        acc_fine_tune=[]
        for user in test_users:
            user_model = self.create_eegnet(self.num_classes)
            user_model.compile(optimizer='adam', loss='binary_crossentropy')
            user_model.load_weights('model_weights/maml/model')
            support_x, support_y = globals.train_data[user]["support"]
            support_x = support_x.reshape(support_x.shape[0], globals.chans, globals.samples, 1)
            support_y      = np_utils.to_categorical(support_y)
            test_data, test_labels = globals.train_data[user]["test"]
            test_data = test_data.reshape(test_data.shape[0], globals.chans, globals.samples, 1)
            test_data, test_labels = tf.convert_to_tensor(test_data), tf.convert_to_tensor(test_labels)
            test_labels = tf.cast(test_labels, tf.int64)

            # Fine-tune on the support set
            user_model.fit(support_x, support_y, epochs=self.fine_tune_steps, verbose=0)

            outputs = user_model(test_data)
            predicted = tf.argmax(outputs, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, test_labels), tf.float32))
            acc_fine_tune.append(accuracy.numpy())
        print(f"Test Accuracy after fine-tuning: {np.mean(acc_fine_tune):.4f}")


