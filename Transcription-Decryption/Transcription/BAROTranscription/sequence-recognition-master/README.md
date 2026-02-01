# Sequence-Based Recognition Models

This repo contains the implementation of various sequence-based recognisers initially
designed for the task of alignment of manuscript ciphers, but now used for recognition
in general.

## Running Experiments

To run an experiment for training or test with a certain configuration you should run
the following command:

```bash
python3 <EXPERIMENT SCRIPT>.py --config_path <PATH TO CONFIG> [--test <PATH TO WEIGHTS>]
```

If you want to write a configuration file, you can run the command:

```bash
python3 <EXPERIMENT SCRIPT>.py --get_template
```

For more detailed help, run:

```bash
python3 <EXPERIMENT SCRIPT>.py --help
```

## Models

- [CTC Loss](/src/seq_alignment/models/cnns.py)
  - VGG Backbone
  - Simple CNN Backbone

## Writing New Experiments or Models

The system incorporates both the implementation of a few models and the training loop
and boilerplate code that make them work. The code is architected as follows

The entry point of the program is an implementation of the ```Experiment``` class. One
must overload the ```initialise_everything``` method to create any objects the
experiment needs and then call ```main``` method to run it.

### Dataloaders

Dataloaders are implemented in the Data folder. This area perhaps needs a little bit
of cleanup as the system right now expects a ```GenericDecryptDataset``` object
whenever the data is needed. In any case, any dataloader that produces a ```Sample```
object when indexed will work fine.

### Formatters

Formatters are classes that convert the output of the model into a **picklable**
alternate representation. This representation may be used for logging purposes or to
compute a specific metric during training. They must be picklable because the results
are accumulated on disk in a pickle file before computing metrics for an entire epoch.
This is needed to implement asynchronous logging and inference.

A formatter must implement the ```__call__``` magic method to develop its main
computation. **The keys this formatter adds on the output dictionary must be provided
by overloading the ```KEYS``` member of the class**.

When more than one formatter is to be applied to the data, the
```formatters.utils.Compose``` class can be used to combine them.

The output of a formatter is a list of dictionaries aligned with the input batch
samples where the keys are the name of the chosen format. Thus, when applying a text
formatter on the output of a model after feeding it a batch of size two will be:

```python
[
    {"text": "<sample text 1>"},
    {"text": "<sample text 2>"}
]
```

Should one use the composition formatter for both text and numbers, the output will
be:

```python
[
    {"text": "<sample text 1>",
     "numbers": "<sample number 1>"},
    {"text": "<sample text 2>",
     "numbers": "<sample number 2>"}
]
```

### Metrics

Metrics are classes that implement the computation of an output metric against the
ground truth. They must implement:
- The ```maximise``` method to assert a higher value of the metric is better or not.
- The ```__call__``` method to perform the computation of the metric itself. This
  computes the value of the metric for a **single** data sample.
- The ```aggregate``` method to combine all sample predictions into a meaningful global
  value.

Metrics also have a ```metrics.utils.Compose``` object to combine multiple of them and
log them in parallel. When used for early stopping and training related purposes, the
first metric that is computed is the one that is taken into account.

### Models

A full model must implement the following methods:
- The ```compute_batch``` method, which generates the output of the model for a batch.
- The ```compute_loss``` method, which computes the loss of the model from the
  previously generated output and the batch information.

This distinction is performed in order to **implement the models by families**. Any CTC
model will compute the loss the same way on the output; therefore any CTC model will
be implemented by inheriting from the CTC base model without requiring a rewrite of
the loss function computation -- only the ```forward``` and ```compute_batch``` methods.

Models have their associate configuration classes which must inherit from the
```BaseConfig``` type. The configuration type should then be added to the class
through the ```MODEL_CONFIG``` member.

## License

This code is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 (see COPYING for
the full license file).
