from MyModel import *
from BinaryClassifier import *


if __name__ == "__main__":
  
  ## testing ##
  myclassifier = Classifier(MyModel(dim_hidden=(2,12),perceptrons_out=10))

  myclassifier.train(num_epochs=30, learning_rate=0.01)
