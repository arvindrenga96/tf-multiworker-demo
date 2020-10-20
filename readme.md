# Tensorflow Distributed Demo

### Contents: 

    1. local-demo.ipynb, A jupyter notebook to demo Tensorflow and Keras
        on a single machine
    2. distributed-demo.py, A python file to demo Tensorflow and Keras 
        on a cluster of machines. 
        
### local-demo.ipynb

A demo for basic image classification using Tensorflow and Keras APIs on a local machine. 
We use the Fashion-MNIST dataset and a simple feed forward dense neural network to 
achieve a classification accuracy of > 86%


### distributed-demo.py
A demo for basic image classification using Tensorflow and Keras APIs on a cluster of
4 virtual machines created on the google cloud platform. We use the MNIST dataset and 
a simple CNN to achieve an accuracy > 85%

### Configuring the cluster

The JSON below is an example cluster configuration. The "cluster" has to be predefined
and static, machines cannot be added or removed once the training process begins. "worker" is a 
list of all worker machines and port numbers that can accept incoming connections. The training does not
begin until all the machines in the cluster come online. The "task" is unique to each worker
where "index" number changes. The worker 0 acts as the cheif machine, does more 
work than the remaining workers in the cluster. 
```json
{
    "cluster": {
        "worker": [
          "localhost:12345",
          "localhost:23456"
        ]
    },
    "task": {
      "type": "worker",
      "index": 0
    }
}
```

### Demo execution instructions

    1. clone the repository on all the worker machines using the following 
    command
    
```shell script
git clone https://github.com/peacekurella/tf-multiworker-demo.git
```
    2. Update the config.json on each machine to have the same cluster info
    with a list of all worker machine IPs and a port number that can accept an 
    incoming connection.
    
    3. Update the config.json on each machine to have a unique index number in the 
    range [0, <number-of-worker-machines>)
    
    4. Start the training process on each machine using the following command

```shell script
python distributed-demo.py --worker <config-index-number>
```