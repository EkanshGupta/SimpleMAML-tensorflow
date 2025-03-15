import numpy as np
import random
from tensorflow.keras.utils import to_categorical

## Helper Functions
def generate_user_data(num_users, num_samples, num_chans, num_time_samples, support_size, query_size):
    """
    Generate data for num_users users with shape (num_samples, num_chans, num_time_samples, 1) and binary labels.
    Each user has 400 samples, with 200 samples per class (0 and 1).
    """
    users_data = {}
    
    for user in range(num_users):
        # Generate K samples per user
        data = np.random.randn(num_samples, num_chans, num_time_samples, 1)
        labels = np.random.randint(0, 2, (num_samples, 1))  # Random binary labels
        labels = to_categorical(labels, num_classes=2)  # Convert to categorical
        
        # Shuffle data and labels
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        
        # Split into support, query, and test sets
        support_data, support_labels = data[:support_size], labels[:support_size]
        query_data, query_labels = data[support_size:support_size + query_size], labels[support_size:support_size + query_size]
        test_data, test_labels = data[support_size + query_size:], labels[support_size + query_size:]
        
        users_data[user] = {
            'support': (support_data, support_labels),
            'query': (query_data, query_labels),
            'test': (test_data, test_labels)
        }
    
    return users_data
