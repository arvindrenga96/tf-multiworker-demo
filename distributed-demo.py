import tensorflow as tf
import os, json
import mnist
import argparse

"""
background:

Collective communications allow us to exchange more easily information across all processes (of a communicator). 
But collective communications are not only useful for this, there are different types of collective communications 
suited for very different objectives[1]. 

"""


if __name__ == "__main__":

    # add the arguments
    parser = argparse.ArgumentParser(description='Run a Multi Worker TF training job')
    parser.add_argument('--worker', default=0, help='process worker number', type=int)
    parser.add_argument('--batch_size', default=64, help='batch size per worker', type=int)
    parser.add_argument('--epochs', default=20, help='number of epochs', type=int)
    parser.add_argument('--steps', default=70, help='number of steps per epoch' ,type=int)

    args = parser.parse_args()

    # clear the config
    # disable GPUs for now
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.pop('TF_CONFIG', None)

    # load the config and set the right worker number
    with open('config.json') as f:
        tf_config = json.load(f)
    tf_config['task']['index'] = args.worker

    # set the config
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    """
    
    This environment variable is used by the distribute startegy to
    get information of about the machines and the processes that are 
    available for training.
    
    """

    # set the batch size and workers
    per_worker_batch_size = args.batch_size
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])
    batch_size = per_worker_batch_size * num_workers

    # set the distribute strategy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    """
    
    The distribute strategy provides the communication policies and data distrubution
    policies for the multi worker training process. 
    
    There are six different distribution strategies[2]:
        1. MirroredStrategy : Synchronous training for multiple devices ( generally GPUs) 
            on the same machine, uses Nvidia's NCCL as communications layer by default
        2. TPUStrategy : Synchronous training using google's TPUs ( dedicated ASICs for deep learning )
            similar to mirrored stratergy, however it uses google's own communications layer
        3. MultiWorkerMirroredStrategy: Synchronous training over multiple devices on multiple machines,
            uses NCCL or gRPC RING for communications based upon the runtime conditions
        4. CentralStorageStrategy: Synchronous training but instead of mirroring variables, all variables
            are placed on the CPU and the operations are mirrored across the local GPUs
        5. ParameterServerStartegy: asynchronous training with very limited support, there are parameter
            servers for variables, computation is replicated across the workers 
        6. OneDeviceStrategy: Variables are places explicitly on one single device, unlike the default strategy
            input is prefetched to the device  
    
    In synchronous training, the cluster would fail if one of the workers fails and no failure-recovery 
    mechanism exists. Using Keras with tf.distribute.Strategy comes with the advantage of fault tolerance in 
    cases where workers die or are otherwise unstable. You do this by preserving training state in the distributed
    file system of your choice, such that upon restart of the instance that previously failed or preempted,
    the training state is recovered. Since all the workers are kept in sync in terms of training epochs and steps, 
    other workers would need to wait for the failed or preempted worker to restart to continue.[3]
      
    """

    # get the dataset
    dataset = mnist.mnist_dataset(batch_size)
    """
    
    This automatically shards the data based on the sharding policy provided 
    by the tf distribute statergy[3]. 
    
    """

    # Model building/compiling need to be within `strategy.scope()`.
    with strategy.scope():
        model = mnist.build_and_compile_cnn_model()

    # fit the model
    model.fit(dataset, epochs=args.epochs, steps_per_epoch=args.steps)


"""

References:
    
    [1]. https://www.codingame.com/playgrounds/349/introduction-to-mpi/introduction-to-collective-communications
    [2]. https://www.tensorflow.org/guide/distributed_training
    [3]. https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    
"""