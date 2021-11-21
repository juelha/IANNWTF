from MyModel import *
from BinaryClassifier import *

def main():

  ## baseline ##
  # initializing the classifier
  baseline = BinaryClassifier(MyModel(dim_hidden=(2,12),perceptrons_out=1))
  # training the model
  baseline.train(num_epochs=10, learning_rate=0.01)

  # results from using test data
  best_acc_baseline = max(baseline.model.test_accuracies)
  loss_baseline = baseline.model.test_losses[-1]

  # results from using validation data
  val_loss_baseline, val_acc_baseline = baseline.evalutate()


  ## optimized ##
  # techniques applied: 
  #   Adam as optimizer, 
  #   kernel_regularizer, used to reduce the sum, l_1 uses abs(x) and l_2 uses square(x)
  #   activity_regularizer, used to reduce the layer's output
  # initializing the classifier
  optimized = BinaryClassifier(
    MyModel(dim_hidden=(2,12), perceptrons_out=1, k_r='l1_l2', a_r='l2'))
  # training the model
  optimized.train(num_epochs=10, learning_rate=0.01, optimizer_func=Adam)


  ## reults ##
  # the results are positive
  # the graphs vary a lot, baseline does not change as much as optimized
  # results from test data
  best_acc_optimized = max(optimized.model.test_accuracies)
  loss_optimized = optimized.model.test_losses[-1]

  # results from validation data
  val_loss_optimized, val_acc_optimized = optimized.evalutate()

  print("\nRESULTS")

  print("\nIMPROVEMENT OVER TESTING DATA")
  print(f'accuracy improvement: {(best_acc_optimized-best_acc_baseline)*100}%')
  print(f'loss improvement:     {(loss_baseline-loss_optimized)*100}%')
  print("\nIMPROVEMENT OVER VALIDATION DATA")
  print(f'accuracy improvement: {(val_acc_optimized-val_acc_baseline)*100}%')
  print(f'loss improvement:     {(val_loss_baseline-val_loss_optimized)*100}%')

  # visualize 
  fig1 = baseline.model.visualize_learning("baseline")
  fig2 = optimized.model.visualize_learning("optimized")

  plt.show()




if __name__ == "__main__":

  main()