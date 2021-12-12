
from BinaryClassifier import *

def main():


    
  # init model
  baseline = BinaryClassifier(LSTM_Wrapper())
  # training the model
  baseline.train(num_epochs=10, learning_rate=0.1,optimizer_func=Adam)

  fig = baseline.trainer.visualize_learning()
  plt.show()




if __name__ == "__main__":

  main()