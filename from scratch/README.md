# From Scratch

Here we use Pytorch to structure our transformer to our liking.

- **Hyper Parameters** are stored in the [parameters.py](parameters.py) file
- The Transformer architecture is in the [model.py](model.py) file
- [data.py](data.py) contains some helper functions to handle the data i.e Tokenize it


## Start The training

After you have Installed the libraries in the requirements.txt file in the parent directory.

Run the following command on your terminal in this directory

This will start the training process.
```sh
  python index.py
```

### Checkpoints & losses

In **line 78**
```py
            torch.save(model.state_dict(), 'model.pt')
```

and **line 85**
```py
 torch.save(losses_dictionary, 'losses.pt')
```

I export the losses and the model to those files, during each epoch of the training.

So you can rename them as you which for a better "directory structure"

