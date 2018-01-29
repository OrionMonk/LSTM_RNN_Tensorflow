## Some basic tensorflow examples
### Fixed sized input without state saving across sequences
The task is to count the number of 1's, modulo 4, in a fixed sized sequential input of length 10. An LSTM cell based RNN with 16 hidden units was used. In this example, states are not saved across successive sequences. 

The results are shown below.


<img src = "https://raw.githubusercontent.com/OrionMonk/RNN_Tensorflow_Examples/master/fixed%20size%20input%20rnn/Image/result.png">

### Fixed sized input with state sharing
The task is to count the number of 1's, modulo 8, upto the current batch. The input format is same as the previous one, except that the labels have been updated with the cumulative sum of 1's across batches rather than individual sums for each input sequence. The size of each input sequence is still 10. States are saved across batches in this example, and is done using a tuple of placeholders. The results are shown below.

<img src="https://raw.githubusercontent.com/OrionMonk/RNN_Tensorflow_Examples/master/fixed%20size%20input%20with%20state%20sharing/image/result.png">

### Variable sized input
The task was to classify whether a name is male or female. The LSTM-RNN was run for 5 epochs and performed quite similar on both the training and the test dataset. In only 5 epochs, the model learned to classify gender of names with a reasonable accuracy of 80 percent on the test dataset, considering the fact that the test names are completely unique and different from training dataset. The training and the test accuracy for a few epochs, along with the prediction on a random sample picked from the test sample is shown below. 

<img src="https://raw.githubusercontent.com/OrionMonk/RNN_Tensorflow_Examples/master/variable%20sized%20input/images/result.png">
