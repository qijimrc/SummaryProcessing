#coding=utf-8
import tensorflow as tf

# config parameters
iterations_per_loop=1000
num_shards=8

# gpu parallel config
num_gpus=1
gpu_order="0 1 2 3"
shard_to_cpu=False
num_async_replicas=1
no_data_parallelism=False
daisy_chain_variables=True
schedule="continuous_train_and_eval"
worker_job="/job:localhost"
worker_id=0
ps_replicas=0
ps_job="/job:ps"
ps_gpu=0
sync=False

# cpu config
num_cpus = 50

# run config
master = ""
evaluation_master=master
model_dir = None
save_summary_steps=100
save_checkpoints_steps = 1000
save_checkpoints_secs=None
keep_checkpoint_max = 20
keep_checkpoint_every_n_hours = 10000
random_seed = None

# session config
graph_options = tf.GraphOptions(
    optimizer_options=tf.OptimizerOptions(
        opt_level=tf.OptimizerOptions.L1,do_function_inlining=False))
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
session_config = tf.ConfigProto(
    allow_soft_placement=True,
    graph_options=graph_options,
    gpu_options=gpu_options,
    log_device_placement=False,
    inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0
)


