from typing import List
import tensorflow as tf
from tensorflow import feature_column
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2
import keras_tuner as kt
import math as m
from pipeline import configs
import datetime

LABEL_KEY = "target"
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32



def _get_hyperparameters() -> kt.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = kt.HyperParameters()
  # Defines search space.
  hp.Choice("f_units", [32, 64, 128])
  hp.Boolean("dropout")
  a = hp.get("f_units")
  b = int(m.log(a,2)) - 1
  hp.Int("num_layers", 1, b)
  hp.Choice('lr', values=[1e-4, 1e-3, 1e-2])
  return hp


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int) -> tf.data.Dataset:
  """Generates features and label for training. Me genera (X,y).

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: schema of the input data.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=LABEL_KEY),
      schema=schema).repeat() 

def _make_keras_model(hparams: kt.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Returns:
    A Keras Model.
  """

  ### encoding de features ###
  feature_columns = []

  features_names = ["score", 
                   "nsepercentil", 
                   "edad", 
                   "antiguedadenbancos", 
                   "mlctc_uva",
                   "antiguedadlaboral"] # No va la variable target.

  for header in features_names:
    feature_columns.append(feature_column.numeric_column(header))

  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)  
  
  

  model = tf.keras.Sequential()
  model.add(feature_layer)
  model.add(tf.keras.layers.Dense(units=hparams.get("f_units"), activation='relu', name= "layer_1"))
  if hparams.get("dropout"):
        model.add(tf.keras.layers.Dropout(rate=0.25))   
  a =  hparams.get("f_units")
  for i in range(hparams.get("num_layers")):
      model.add(tf.keras.layers.Dense(units= int(a/(2**(i + 1))), activation="relu", name= f"layer_{i + 2}"))
      model.add(tf.keras.layers.BatchNormalization())         
  model.add(tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer"))

  learning_rate = hparams.get("lr")

  optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
  model.compile(
              optimizer= optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.AUC(name="auc"),
              tf.keras.metrics.Precision(),
              tf.keras.metrics.Recall()])
    
  
  return model

def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  # RandomSearch is a subclass of keras_tuner.Tuner which inherits from
  # BaseTuner.
  tuner = kt.BayesianOptimization(
    _make_keras_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=2,
    hyperparameters=_get_hyperparameters(),
    executions_per_trial=1,
    overwrite=True,
    directory=configs.TUNER_DIR, #detailed logs, checkpoints, etc.
    project_name="tuner_test"
)

  schema = tfx.utils.parse_pbtxt_file(fn_args.schema_path, schema_pb2.Schema())   

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      batch_size=TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      batch_size=EVAL_BATCH_SIZE)

  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5)

  return tfx.components.TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'epochs': 70,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps,
          'callbacks':[stop_early]
      })



# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  # This schema is usually either an output of SchemaGen or a manually-curated
  # version provided by pipeline author. A schema can also derived from TFT
  # graph if a Transform component is used. In the case when either is missing,
  # `schema_from_feature_spec` could be used to generate schema from very simple
  # feature_spec, but the schema returned would be very primitive.
  schema = tfx.utils.parse_pbtxt_file(fn_args.schema_path, schema_pb2.Schema())

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      batch_size=TRAIN_BATCH_SIZE)
  
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      batch_size=EVAL_BATCH_SIZE)
  
  hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
  
  ### TB ###
  logs_dir = "gs://hf-exp/vpoc/mpg/tuner/tfx/tboard/" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S") + "tuner"
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

  model = _make_keras_model(hparams)
  model.fit(
      train_dataset,
      epochs=2,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  # The result of the training should be saved in `fn_args.serving_model_dir`
  # directory.
  model.save(fn_args.serving_model_dir, save_format='tf')
