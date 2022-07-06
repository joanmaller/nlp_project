# RNN lyrics generator
## Introduction
To recreate the style of specific writers, I make the use of a Recurrent Neural Network trained only on the corpus of the specific artist. 


## Requirements

We need tensorflow, numpy for training and testing the RNN and matplotlib for plotting the model accuracy and loss plots.


```bash
pip install tensorflow
pip install numpy
pip install matplotlib
```

## Usage

The way to train a new model, is by executing the [train_rnn](train_rnn.py) python script followed by a text data file, in this case the lyrics of [leonard-cohen](leonard-cohen.txt).  
```bash
python train_rnn.py leonard-cohen.txt
```
After the script is run, we can locate a .h5 model file in the folder to load it whenever we want.

To test the model, we will use the [test_rnn](test_rnn.py) python script, followed by the data text file, and the model file path as second argument. 

```bash
python test_rnn.py leonard-cohen.txt leonard-cohen.h5

```


Another way of testing the model is through a [Huggingface Spaces](https://huggingface.co/spaces/jmaller/rnn-amywinehouse) where it is run on Gradio and displayed for everyone.

![Image of online demo](Demo_Amy.JPG "" )



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
