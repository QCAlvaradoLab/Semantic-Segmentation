Setup instructions in Linux

Install necessary libraries.

```pip install -r requirements.txt```

Look into `train_config2.json` as it acts as the template for the configuration file of the project. 

To run the training, use:

```
python main.py
```

To test a trained model, use:

```
python main.py --test --model <PATH>
```

`training.log` will contain the log for the epochs of training as well as other information. 

During the course of the training, the code accesses and modifies parameters of the master config file provided in order to persist essential information like best epoch based on validation loss. More notably, it also stores other stats like epoch to resume from and model file to load upon re-execution with same config file to auto-resume a training loop terminated for some reason. Essentially, every unique project can be managed with a separate config file. 
