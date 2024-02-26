# PlantTraits2024 - FGVC11
## Description 
This competition aims to predict plant properties - so called plant traits - from citizen science plant photographs. Why are plant traits currently so relevant? **Plant traits are plant properties that are used to describe how plants function how they interact with the environment**. For instance, the trait of plant canopy height indicates how good a plant is at overshadowing its neightbors in the competition for sun light. Robust leaves (indicated by the leaf mass pear leaf area) indicate that plants optimize towards extreme conditions, such as heavy winds or droughts. Yet, environmental conditions are not static. Due to global change, the **biosphere is being transformed at accelerating pace**. Especially climate change is assumed to drastically impact the functioning of the ecosystems. This includes several processes, e.g. adaptions of plants and their traits to new conditions or even a altered plant species distribution with a resulting modification of the distribution of plant traits. However, **we can hardly project on a global scale how plant traits and as such entire ecosystems will react to climate change because we do not have sufficient data on plant traits**.

A **data treasure** in this regard may be the growing availability of **citizen science photographs**. Thousands of citizens around the globe photograph plants with species identification apps (examples are iNaturalist or Pl@ntNet). The species are identified using AI algorithms, and the prediction, photograph, and geolocation are curated in open databases. There are already more than **20 million plant photographs available, covering all ecosystem types and continents**.

<img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3319247%2Fec99f4290e7c7ff3cdc5fe57ed9c1509%2Fsample_images.png?generation=1676369779907551&alt=media'>

In its original form, this data initially only provides information on the species name of a plant and not its traits. However, a pioneering study showed that **artificial intelligence can predict plant traits from such photographs using Convolutional Neural Networks** (Schiller et al., 2021). To achieve this, we paired sample images from the iNaturalist database with plant trait data that scientists have been curating for decades for various species. The challenge was that the images and plant trait observations were not acquired for the same plant individuals or at the same time. Nevertheless, using a weakly supervised learning approach, we trained models that demonstrated the potential of this approach for a few plant traits. However, this potential was evident only for a limited number of plant traits and a couple of thousand images. **This competition aims to further unlock the potential of predicting plant traits from plant photographs**. To achieve this, we gathered more training data (over 30,000 images with labels).

Find here the original article:
* *Schiller, C., Schmidtlein, S., Boonman, C., Moreno-Martínez, A., & Kattenborn, T. (2021). Deep learning and citizen science enable automated plant trait predictions from photographs. Scientific Reports, 11(1), 16395.*
https://www.nature.com/articles/s41598-021-95616-0

<img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3319247%2F4df028d3fc43ea0fade8833e14671360%2Fsample_preds.png?generation=1676369823384723&alt=media'>

The interested reader may also see these references for some background and the general idea:
* *Wolf, S., Mahecha, M. D., Sabatini, F. M., Wirth, C., Bruelheide, H., Kattge, J., … & Kattenborn, T. (2022). Citizen science plant observations encode global trait patterns. Nature Ecology & Evolution, 1-10.*
https://www.nature.com/articles/s41559-022-01904-x
* *Moles, A.T., Xirocostas, Z.A. Statistical power from the people. Nat Ecol Evol 6, 1802–1803 (2022).*
https://www.nature.com/articles/s41559-022-01902-z

## Installation
### Clone this repository
```bash
git clone https://github.com/farrosalferro/PlantTraits2024---FGVC11.git
cd PlantTraits2024---FGVC11
conda env create -f environment.yml
conda activate planttraits2024
```

### Download the dataset
Download the dataset from the competition page [here](https://www.kaggle.com/competitions/planttraits2024/data) and extract it to the `data` (you have to create it first) folder.

or use the kaggle api to download the dataset
```bash
mkdir data
cd data
kaggle competitions download -c planttraits2024
unzip planttraits2024.zip
```

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │   ├── test_images
  │   ├── train_images
  │   ├── sample_submission.csv
  │   ├── target_name_meta.tsv
  │   ├── test.csv
  │   └── train.csv
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      ├── submission.py
      └── ...
  ```
## Usage
### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "PlantTraitsModel_ViTb_Dense",
    "n_gpu": 1,
    "arch": {
        "type": "PlantTraitsModel_ViTb_Dense",
        "args": {
            "input_dim": 163,
            "num_classes": 6,
            "image_output_dim": 64,
            "tabular_output_dim": 64,
            "hidden_dim": [
                64,
                128,
                256,
                128,
                64
            ],
            "dropout": 0.5
        }
    },
    "data_loader": {
        "type": "PlantTraitsDataLoader",
        "args": {
            "data_dir": "data/",
            "batch_size": 32,
            "img_size": 224,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 4,
            "training": "True"
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "amsgrad": true
        }
    },
    "loss": "R2Loss",
    "metrics": [
        "R2Metrics"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.5
        }
    },
    "trainer": {
        "epochs": 2,
        "save_dir": "saved/",
        "regularization": 0.4,
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10,
        "tensorboard": true
    }
}
```
Add addional configurations if you need.
### Using config files
Modify the configurations in `.json` config files, then run:
  ```
  python train.py --config config.json
  ```
### Resuming from checkpoints
You can resume from a previously saved checkpoint by:
  ```
  python train.py --resume path/to/checkpoint
  ```
### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```
## Implementation Details
### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 
### Custom CLI options
Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.
This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.
  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.
### Model
You can create you own model by add new class `YourModelName(BaseModel)` inside the `model/model.py`. The Basemodel is inherited from `torch.nn.Module` and modified native function such as `__str__` function to prints the number of trainable parameters.

Make sure to give the output in form of `Dict` with `head` and `aux_head` as its keys where the initial and the latter contain the main prediction (`_mean`) and auxiliary prediction (`_std`). Please refer to `PlantTraitsModel_ViTb_Dense(BaseModel)` model as an **example**.

Do not forget to change the `config.json` to include the name of your model and its arguments.

### Metrics
The models will be evaluated against the independent test data. The evaluation metric for this competition is the mean R2 over all 6 traits. The R2 is commonly used for evaluating regression models and is the ratio of the sum of squares the residuals (SSres) to the total sum of squares (SStot).

$$
\begin{aligned}
SS_{residual} &= \sum_i(y_i - f_i)^2 \\
SS_{total} &= \sum_i(y_i - \bar{y})^2 \\
\text{finally the } R^2 \text{score:} \\
R^2 &= 1 - \frac{SS_{residual}}{SS_{total}}
\end{aligned}
$$

Where $f$ is the predicted value, $y$ and $\bar{y}$ are the ground truth and its mean, respectively.The R2 can result in large negative values. To prevent that we will only consider R2 values > 0. The implementation is located in `model/metric.py`.

### Loss
As we want to maximize the metrics while minimizing the loss, we take the substraction part of the metrics as the loss:
$$
loss = \frac{SS_{residual}}{SS_{total}}
$$

The loss is implemented in `model/loss.py`. 

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:
  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```
### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument and save the prediction by `--submit_filename`:

```
python test.py --configs /path/to/config --resume /path/to/checkpoint --submit_filename submission.csv
```
the submission file will be saved inside the `submission/exp_name/` folder where `exp_name` is the value of `name` key inside the `config.json`

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).
**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`
### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```
The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.
A copy of config file will be saved in the same folder.
**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```
### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).
1. **Install**
    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.
    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).
2. **Run training** 
    Make sure that `tensorboard` option in the config file is turned on.
    ```
     "tensorboard" : true
    ```
3. **Open Tensorboard server** 
    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`
By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 
**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## TODOs
- [] Add WandB logger
- [] Add feature for running using slurm
## License
This project is licensed under the MIT License. See  LICENSE for more details
## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95), [Pytorch Template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque), and [PlantTraits2024: KerasCV Starter Notebook](https://www.kaggle.com/code/awsaf49/planttraits2024-kerascv-starter-notebook) by [Awsaf](https://www.kaggle.com/awsaf49).