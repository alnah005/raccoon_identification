cd transferLearning
cd models
cd research
PIPELINE_CONFIG_PATH=/home/fortson/alnah005/transferLearning/md_v4.1.0.config
MODEL_DIR=/home/fortson/alnah005/transferLearning/mymodel/
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=6
python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} --alsologtostderr