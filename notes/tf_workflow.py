
'worflow start to finish'


# source: https://www.youtube.com/watch?v=2FmcHiLCwTU
# new source: https://github.com/Spinkk/TeachingTensorflow/blob/main/basics/Automatic%20Differentiation%20with%20tf.GradientTape().ipynb 
# prob w linux: https://stackoverflow.com/questions/63177156/tensorflow-dataloading-issue 


'1. LOAD MNIST DATA'
import tensorflow_datasets as tfds
import tensorflow as tf
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)



'2. DATA PIPELINE'
"""
The general structure of a data pipeline looks something like this:

    Create the dataset from a source
    Apply all preprocessing steps, parallelize wherever possible!
    Optionally cache the preprocessing
    Shuffle then batch, then prefetch
S: https://colab.research.google.com/drive/1wzU6IwHZhEjuy3P6Wa6nwrYX5OVRe0lJ#scrollTo=kcrghVCT6AX- 
"""
def prepare_mnist_data(mnist):

    # PREPROCESSING 
    #flatten the images into vectors
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    #convert data from uint8 to float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
    #create one-hot targets
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    
    # CACHING (opt)
    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    mnist = mnist.cache()

    # SHUFFLE, BATCH, PREFETCH
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(8)
    mnist = mnist.prefetch(20)

    #return preprocessed dataset
    return mnist

train_ds = train_ds.apply(prepare_mnist_data)
test_ds = test_ds.apply(prepare_mnist_data)



'2. set paramters'
learning_rate = 0.3 # defines how fast we want to update our weights
                    # if too large: model might skip optimal solution 
                    # if too small: need too many epochs 
training_iteration = 30
batch_size = 100
display_step = 2


'3. building operations'
# TF graph input
# flattening image for more efficent formatting

x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
                                         # 784 = dimensions of a single flattened mnist image
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes


'4. Create a model'

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# create a name scopes
# More name scopes will clean up graph representation
# scopes help us organize nodes in the graph visualizer called tensor board which we will view at the end
# 1. scope: implementing model
with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

  
# Add summary ops (operations) to collect data (later visualize the distribution of weights and biases)
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# 2. scope: create cost function
with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)

# 3. scope: training the model
with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()


'5. training and visu'
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    
    
    # Change this to a location on your computer
    summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per iteration step
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning completed!")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))