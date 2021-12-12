
from BinaryClassifier import *

def main():

  ## baseline ##
  # initializing the classifier
  baseline = BinaryClassifier()
  # training the model
  baseline.train(num_epochs=10, learning_rate=0.01)



if __name__ == "__main__":

  main()