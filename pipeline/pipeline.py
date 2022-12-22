from typing import Dict, List, Optional
import os
from tfx import v1 as tfx
from tfx.proto import example_gen_pb2


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    query_train: str,
    query_test: str,
    run_fn: str,
    region:str,
    tuner_fn: str,
    tuner_dir: str ,
    serving_model_dir: str,
    ai_platform_training_args: Optional[Dict[str, str]] = None,
    beam_pipeline_args: Optional[List[str]] = None
) -> tfx.dsl.Pipeline:

  # ExampleGen Component
  
  input_config = example_gen_pb2.Input(splits=[
                 example_gen_pb2.Input.Split(name='train', pattern=query_train),
                 example_gen_pb2.Input.Split(name='eval', pattern=query_test)
                               ])

  example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
      input_config=input_config
      )
  
  # StatisticsGen Component

  statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

  
  # SchemaGem Component

  schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'])
  
  # Tuner Component
  
  tuner = tfx.extensions.google_cloud_ai_platform.Tuner(
        tuner_fn=tuner_fn,
        examples=example_gen.outputs['examples'],
        schema = schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=20),
        eval_args=tfx.proto.EvalArgs(num_steps=5),
        custom_config = {
        tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
        tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
        tfx.extensions.google_cloud_ai_platform.experimental.TUNING_ARGS_KEY:
            ai_platform_training_args,
        tfx.extensions.google_cloud_ai_platform.experimental.REMOTE_TRIALS_WORKING_DIR_KEY:
                  os.path.join(tuner_dir, 'trials')    
                        }
        )

  # Trainer Component

  trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
       run_fn = run_fn,
       examples = example_gen.outputs['examples'],
       schema = schema_gen.outputs['schema'],
       hyperparameters=tuner.outputs['best_hyperparameters'],
       train_args=tfx.proto.TrainArgs(num_steps=20),
       eval_args=tfx.proto.EvalArgs(num_steps=5),
       custom_config = {
        tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
        tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: region,
        tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
            ai_platform_training_args
                        }
                        )
                        
 
  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      push_destination=tfx.proto.PushDestination(
      filesystem=tfx.proto.PushDestination.Filesystem(
      base_directory=serving_model_dir)))
  
  components = [
      example_gen,
      statistics_gen,
      schema_gen,
      tuner,
      trainer,
      pusher
            ] 

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      beam_pipeline_args=beam_pipeline_args)
