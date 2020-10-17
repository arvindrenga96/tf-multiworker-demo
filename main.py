import tensorflow as tf
import os, json
import mnist
import argparse


if __name__ == "__main__":

    # add the arguments
    parser = argparse.ArgumentParser(description='Run a Multi Worker TF training job')
    parser.add_argument('--worker', default=0, help='process worker number', type=int)
    parser.add_argument('--batch_size', default=64, help='batch size per worker', type=int)
    parser.add_argument('--epochs', default=5, help='number of epochs', type=int)
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

    # set the batch size and workers
    per_worker_batch_size = args.batch_size
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])
    batch_size = per_worker_batch_size * num_workers

    # set the distribute stratergy
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # get the dataset
    dataset = mnist.mnist_dataset(batch_size)

    # Model building/compiling need to be within `strategy.scope()`.
    with strategy.scope():
        model = mnist.build_and_compile_cnn_model()

    # fit the model
    model.fit(dataset, epochs=args.epochs, steps_per_epoch=args.steps)