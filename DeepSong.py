#We will import and download the required packages and dependencies in this chunck of code
import tensorflow as tf #To create our NN
import numpy as np #For mathematical facilities
import os #For computer related methods
import time #For time handling
import functools #For higher order functions
from IPython import display as ipythondisplay 
from tqdm import tqdm #For loading bars

#Installing MIT data set
!pip install mitdeeplearning #Data set install and import
import mitdeeplearning as mdl
#abcmidi timidity is to run abc notation without the need of a synthesizer
!apt-get install abcmidi timidity > /dev/null 2>&1

#Now we are going to load the songs from the data set
songs = mdl.lab1.load_training_data()

#We run the first element of the songs iterable to analyze the data set
example_song = songs[2]
print("\nExample song: ")
print(example_song)

# Convert the ABC notation to audio file and listen to it
mdl.lab1.play_song(example_song)

#Joining every song loaded in the "songs" data set
songs_joined = "\n\n".join(songs) # A \n will seperate every main category
                                  #This categories being X,T,Z...
#Setting and sorting
vocab=sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")

### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d", 
# We can evaluate `char2idx["d"]`.  
char2idx = {u:i for i, u in enumerate(vocab)}
# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char={u:i for u,i in enumerate(vocab)}

#We can demonstrate that we have mapped it correctly with a sliced dictionary
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

'''TODO: Write a function to convert the all songs string to a vectorized
    (i.e., numeric) representation. Use the appropriate mapping
    above to convert from vocab characters to the corresponding indices.

  NOTE: the output of the `vectorize_string` function 
  should be a np.array with `N` elements, where `N` is
  the number of characters in the input string
'''

def vectorize_string(string):
  vector_list=[]
  for i in string:
    vector_list.append(char2idx[i])
  vectorized_np=np.array(vector_list)
  return vectorized_np

vectorized_songs = vectorize_string(songs_joined)
print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"

### Batch definition to create training examples ###

def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string minus 1
  #the one is explained do to the shift between input and output
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  #idx will be of length "batch_size"
  idx = np.random.choice(n-seq_length, batch_size)

  '''TODO: construct a list of input sequences for the training batch'''
  input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
  '''TODO: construct a list of output sequences for the training batch'''
  output_batch = [vectorized_songs[i+1:i+seq_length+1] for i in idx]

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch

  # Provided code to test our function 
test_args = (vectorized_songs, 10, 2)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_next_step(get_batch, test_args): 
   print("======\n[FAIL] could not pass tests")
else: 
   print("======\n[PASS] passed all tests!")

def LSTM(rnn_units): #We first define our type of LSTM cell, to later call
#the function when we need it
  return tf.keras.layers.LSTM(
    rnn_units, #Number of units or cells
    return_sequences=True, 
    recurrent_initializer='glorot_uniform', #Parameter preferences
    recurrent_activation='sigmoid', #Parameter preferences
    stateful=True,
  )

'''TODO: Add LSTM and Dense layers to define the RNN model using the Sequential API.'''
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors 
    #   of a fixed embedding size, basically our input
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units. 
    # TODO: Call the LSTM function defined above to add this layer.
    LSTM(rnn_units),
    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size. 
    # TODO: Add the Dense layer.
    tf.keras.layers.Dense(vocab_size) #Basically our output
  ])

  return model

# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

@tf.function
def step(x,y):
  with tf.GradientTape(persistent=True) as tape:
      y_pred=model(x)
      loss=tf.keras.losses.sparse_categorical_crossentropy(y,y_pred, from_logits=True)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

def TrainModel(model,optimizer):
  history = []
  plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
  if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

  for iter in tqdm(range(num_training_iterations)):
    
    #Train
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    loss = step(x_batch, y_batch)

    # Update the progress bar
    history.append(loss.numpy().mean())
    plotter.plot(history)

    # Update the model with the changed weights!
    if iter % 100 == 0:     
      model.save_weights(checkpoint_prefix)

  # Save the trained model and the weights
  model.save_weights(checkpoint_prefix)

def GenerateSong(model,start_string,length):
  #We define our input string
  input_eval= [char2idx[char] for char in start_string]
  #We exapnd the dimensions of our list to accomodate with tensorlfow
  input_eval=tf.expand_dims(input_eval,0)

  generated_song=[]

  # Here batch size == 1
  model.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(length)):
    predictions=model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    generated_song.append(idx2char[predicted_id])
  return (start_string + ''.join(generated_song))

model=build_model(vocab_size,embedding_dim,rnn_units,batch_size)
optimizer=tf.keras.optimizers.Adam(learning_rate)
TrainModel(model,optimizer)

model = build_model(vocab_size,embedding_dim,rnn_units, batch_size=1)
# Restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

generated_abc=GenerateSong(model, "X", 1000)
print(generated_abc)

generated_songs=mdl.lab1.extract_song_snippet(generated_abc)
for i, song in enumerate(generated_songs): 
  # Synthesize the waveform from a song
  waveform = mdl.lab1.play_song(song)

  # If its a valid song (correct syntax), lets play it! 
  if waveform:
    print("Generated song", i)
    ipythondisplay.display(waveform)