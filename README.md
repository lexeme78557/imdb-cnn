# imdb-cnn

A PyTorch implementation of a CNN to be used for classifying [IMDb movie ratings](https://pytorch.org/text/stable/datasets.html#imdb) as "positive" or "negative".

## How to Run 

To change up parameters such as learning rate, batch size, etc, please change them within main.py

The model will check to see if there is a GPU available. If not, be warned that CPU is significantly slower and will take a while. 80% accuracy may be achieved within 10 epochs for individuals looking for a shorter dive.

```
python main.py
```


