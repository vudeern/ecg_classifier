

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

"""## Function to plot loss
Same function we have been using
"""

##########
# Plot the loss as a function of epoch.  Can use this plot to determine how the
# training went.
def plot_loss( epochs, loss, label, val=None ):

# Make a figure and add axes labels
    plt.figure()
    plt.xlabel( 'Epoch' )
    plt.ylabel( label )

# Just plot the loss vs the epoch number
    plt.plot( epochs, loss )
    if val != None:
        plt.plot( epochs, val )
    plt.show()
#
# End of plot_loss function
##########

"""## Load in the ECG data from a file
All the ECG data is stored in a .csv file that I share with you from my Google Drive.  In order to load the data from it, you will download it to your Colab session and then open it from there.
"""

# Location of file
url = 'https://drive.google.com/file/d/1Iw6QXzsxB8nLHBmXTheJsGa1BfqT_0nr/view?usp=sharing'

# Construct URL to download the file
file_id = url.split('/')[-2]

# Download the file to a local temporary path - using the gdown library
import gdown
local_file_path = 'ecg_data.csv'
gdown.download( id=file_id, output=local_file_path, quiet=False )

# Open the downloaded local file and store the data in lines
with open( local_file_path ) as file:
  lines = file.readlines()

"""## Parse each line of ECG data
The last piece of data on each line is the label.

The rest of the data on each line is the ECG signal collected at 300 Hz.  The original data was of variable length, but for this project, each line contains exactly 2000 data points from the ECG.

### All we did here was take each of the 2000 data points and made them each a feature for our model.
"""

# Set up a list of features and list of corresponding labels
features = []
labels = []
for line in lines:

# We know its a .csv file, so split each line apart by commas
    pieces = line.split(',')

# The last piece of data on each line is the label (0,1,2,3)
# Convert it to an integer because it is still a string after the split command
    labels.append( int( pieces[-1].strip() ) )

# Split out the ECG data (don't include the last column)
# At this poitnt each piece is still treated as a string, so it has to be converted to a floating point number
    string_data=pieces[0:-1]

# Store all the data for each case in a list
    data = []
    for piece in string_data:
        data.append( float( piece ) )

# Store the ECG data as a tensor in the features list
    features.append( tf.convert_to_tensor( data ) )

"""## (optional) Look at the ECG data
If you want to check an ECG signal and display the label. You can set EXAMPLE to whatever number ECG trace to look at.
"""

EXAMPLE = 12
plt.figure()
plt.plot( features[EXAMPLE] )
plt.show()
print( labels[EXAMPLE] )
print( len( features[EXAMPLE] ) )

"""## Split the data up into a training set and a validation set
Had to use convert to tensor function to convert the input for the model from a list to a tensor.
"""

# Split the data into training and validation
from sklearn.model_selection import train_test_split
train_features, val_features, train_labels, val_labels = train_test_split( features, labels, test_size=0.2 )
train_features = tf.convert_to_tensor( train_features )
val_features = tf.convert_to_tensor( val_features )

# To see how much of each data set is classified as what label [0,1,2,3]
# Especially useful to see how much of the data is classified as normal
bc = np.bincount( train_labels )
print( "training:", bc/sum(bc) )
bc = np.bincount( val_labels )
print( "validation:", bc/sum(bc) )

"""## Convert the output categories from a single integer value into one-hot vectors
The result of the IntegerLookup layer will be each of the classifications (0,1,2,3) plus a -1 which indicates 'Unknown'.  We want to keep the unknown because it provided robustness. if there is some bad testing data, the model will be able to handle it.

Encode both the training and testing labels as one-hot vectors
"""

# Set up one hot vector for output
categorizer = keras.layers.IntegerLookup( output_mode = 'one_hot' )
categorizer.adapt( train_labels )

# See all the labels that the categorizer found
print( categorizer.get_vocabulary() )

# Now encode both the training and testing labels as one-hot vectors
encoded_train_labels = categorizer( train_labels )
encoded_val_labels = categorizer( val_labels )



##########
# Function to pre-process features
def pre_process_features( input_features ):

# Do any custom pre-processing of the features here - (optional) TO DO
    output_features = input_features

# If you use convolutional layers, you must add a dimension to the features
# If you don't use a convolutional layer first, you need to comment out this line
    output_features = tf.convert_to_tensor( np.reshape( output_features, ( ( output_features.shape[0], output_features.shape[1], 1 ) ) ) )

    return output_features
#
# End of pre_process_features function
##########

"""## Build and compile the model"""

##########
# Build and compile the model - TO DO
def build_compile_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(2000, 1)))

    model.add(keras.layers.Conv1D(64, 15, activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Dropout(0.10))

    model.add(keras.layers.Conv1D(128, 9, activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Dropout(0.20))

    model.add(keras.layers.Conv1D(128, 5, activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Dropout(0.15))

    model.add(keras.layers.Conv1D(128, 3, activation='relu', padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.20))

    model.add(keras.layers.GlobalAveragePooling1D())

    model.add(keras.layers.Dense(96, activation='relu'))
    model.add(keras.layers.Dropout(0.30))
    model.add(keras.layers.Dense(48, activation='relu'))
    model.add(keras.layers.Dropout(0.15))

    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(
        keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    return model
#
# End of build_compile_model function
##########

"""## Train the model"""

##########
# Train the model - TO DO
def train_model( model, train_features, train_labels, val_features = None, val_labels = None ):

    early_stop = keras.callbacks.EarlyStopping( monitor='val_categorical_accuracy', mode='max', patience=10, restore_best_weights=True )

    reduce_lr = keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001 )

    history = model.fit( train_features, train_labels, batch_size=16, epochs=120, validation_data=(val_features, val_labels), callbacks=[early_stop, reduce_lr] )
    return history
#
# End of train_model function
##########

"""## Run the functions to pre-process data, build, compile, and train the model"""

# Build and compile the model
model = build_compile_model()

# Run the pre-processing of features on both the training and validation data
train_features = pre_process_features( train_features )
val_features = pre_process_features( val_features )

# Train
history = train_model( model, train_features, encoded_train_labels, val_features, encoded_val_labels )

"""## View the results"""

model.summary()
plot_loss( history.epoch, history.history['loss'], 'Loss', history.history['val_loss'] )
plot_loss( history.epoch, history.history['categorical_accuracy'], 'Accuracy', history.history['val_categorical_accuracy'] )



from google.colab import drive
drive.mount('/content/gdrive')


output_fname = '/content/gdrive/MyDrive/Colab Notebooks/ML_Assignment_ECG.keras'
model.save( output_fname )


model_fname = output_fname
saved_model = keras.models.load_model( model_fname )

# Load in test data here
# test_features = []
# encoded_test_labels = []

#encoded_test_labels = categorizer( test_labels )
#results = saved_model.evaluate( np.asarray( test_features), np.asarray( encoded_test_labels) )