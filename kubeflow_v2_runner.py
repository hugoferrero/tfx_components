import os
from absl import logging
from pipeline import configs
from pipeline import pipeline
from tfx import v1 as tfx



_OUTPUT_DIR = os.path.join('gs://', configs.GCS_OUTPUTS)

_PIPELINE_ROOT = os.path.join(_OUTPUT_DIR, 'tfx_pipeline_output',
                              configs.PIPELINE_NAME)
_SERVING_MODEL_DIR = os.path.join(_PIPELINE_ROOT, 'serving_model')




def run():
  """Define a pipeline to be executed using Kubeflow V2 runner."""

  runner_config = tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(
      default_image=configs.PIPELINE_IMAGE)

  dsl_pipeline = pipeline.create_pipeline(
      pipeline_name=configs.PIPELINE_NAME,
      pipeline_root=_PIPELINE_ROOT,
      query_train=configs.QUERY_TRAIN,
      query_test=configs.QUERY_TEST,
      run_fn=configs.RUN_FN,
      region=configs.GOOGLE_CLOUD_REGION,
      tuner_fn=configs.TUNER_FN,
      tuner_dir=configs.TUNER_DIR,
      serving_model_dir=_SERVING_MODEL_DIR,
      ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
      beam_pipeline_args=configs.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS
                                         )

  runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=runner_config,
    output_dir=_PIPELINE_ROOT,
    output_filename=configs.PIPELINE_NAME + '_pipeline.json' 
    )
   

  runner.run(pipeline=dsl_pipeline)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
