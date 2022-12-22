"""TFX  emplate configurations.
This file defines environments for a TFX pipeline.
"""

import os  # pylint: disable=unused-import

PIPELINE_NAME = 'tfx-mdhc-v21'

# GCP related configs.

# Following code will retrieve your GCP project. You can choose which project
# to use by setting GOOGLE_CLOUD_PROJECT environment variable.
try:
  import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  try:
    _, GOOGLE_CLOUD_PROJECT = google.auth.default(quota_project_id = 'my_project')
  except google.auth.exceptions.DefaultCredentialsError:
    GOOGLE_CLOUD_PROJECT = 'my_project'
except ImportError:
  GOOGLE_CLOUD_PROJECT = 'my_project'

# Specify your GCS bucket name here. You have to use GCS to store output files
# when running a pipeline with Kubeflow Pipeline on GCP or when running a job
# using Dataflow. Default is '<gcp_project_name>-kubeflowpipelines-default'.
# This bucket is created automatically when you deploy KFP from marketplace.

GCS_BUCKET_NAME = 'hf-exp/vpoc/mdhc'
GCS_OUTPUTS = 'hf-exp/vpoc/mdhc/outputs'
GOOGLE_CLOUD_REGION = 'us-east4'
# BQ Constants
BQ_DATASET_NAME="my_dataset"
BQ_ML_TABLE_NAME="my_table"
BQ_URI_SIN_PREFIX = f"{GOOGLE_CLOUD_PROJECT}.{BQ_DATASET_NAME}.{BQ_ML_TABLE_NAME}"
# Following image will be used to run pipeline components run if Kubeflow
# Pipelines used.
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'
RUN_FN = 'models.keras_model.model.run_fn'
TUNER_FN = 'models.keras_model.model.tuner_fn' 
TUNER_DIR = 'gs://hf-exp/vpoc/mpg/tuner/tfx/tuner_logs'

### Queries ###
QUERY_TRAIN = f"""SELECT score, 
               nsepercentil, 
               edad, 
               antiguedadenbancos, 
               mlctc_uva,
               antiguedadlaboral, 
               target FROM `{BQ_URI_SIN_PREFIX}`
               where periodo = 202009"""

QUERY_TEST = f"""SELECT score, 
               nsepercentil, 
               edad, 
               antiguedadenbancos, 
               mlctc_uva,
               antiguedadlaboral, 
               target FROM `{BQ_URI_SIN_PREFIX}`
               where periodo = 202010"""


BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
    '--project=' + GOOGLE_CLOUD_PROJECT,
    '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp')
    ]

                    

GCP_AI_PLATFORM_TRAINING_ARGS = {
     'project': GOOGLE_CLOUD_PROJECT,
     'worker_pool_specs': [{
     'machine_spec': {
     'machine_type': 'n1-standard-4',
                      },
     'replica_count': 1,
     'container_spec': {
     'image_uri': PIPELINE_IMAGE
                        }
                          }]
                               } 