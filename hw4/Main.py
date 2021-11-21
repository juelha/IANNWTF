from MyModel import *
from BinaryClassifier import *


if __name__ == "__main__":

  ## testing ##
  myclassifier = BinaryClassifier(MyModel(dim_hidden=(2,12),perceptrons_out=10))

  myclassifier.train(num_epochs=10, learning_rate=0.01)

  best_acc = max(myclassifier.model.test_accuracies)

  print(f'best accuracy: {best_acc}')
