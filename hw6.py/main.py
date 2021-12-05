import matplotlib.pyplot as plt
from MyClassifier import *

###################################################
## 3 Training & Analysis                         ##
###################################################
def main():

  ## MyResNet ##
  # initializing the classifier
  classifier_res= MyClassifier(model = MyResNet())
  # training
  classifier_res.train(num_epochs=1,learning_rate=1)


  ## MyDenseNet ##
  # initializing the classifier
  classifier_dense = MyClassifier(model = MyDenseNet())
  #training
  classifier_dense.train(num_epochs=1,learning_rate=1)


  ## RESULTS ##
  print("\nNETWORKS")
  print(classifier_res.model.summary())
  print(classifier_dense.model.summary())

  print("\nRESULTS")

  print("\nIMPROVEMENT OVER TESTING DATA")
  print("RESNET")
  print(f'accuracy: {max(classifier_res.model.test_accuracies)*100}%')
  print(f'loss:     {classifier_res.model.test_losses[-1]*100}%')
  print("DENSENET")
  print(f'accuracy: {max(classifier_dense.model.test_accuracies)*100}%')
  print(f'loss:     {classifier_dense.model.test_losses[-1]*100}%')

  # visualize 
  fig1 = classifier_res.model.visualize_learning("classifier_res")
  fig2 = classifier_dense.model.visualize_learning("classifier_dense")

  plt.show()


  print("okay")




if __name__ == "__main__":

  main()

  # testing
  # initializing the classifier
  #classifier_dense = MyClassifier(model = MyDenseNet())
  #training
 # classifier_dense.train(num_epochs=1,learning_rate=1)





