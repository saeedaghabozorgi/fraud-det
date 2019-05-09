# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import pandas as pd
import tensorflow as tf


INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"

my_feature_columns=[]

def model_fn(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)




def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset



def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset



#accepts inference requests and prepares them for the model. 
#The function returns a tf.estimator.export.ServingInputReceiver object, which packages the placeholders and the resulting feature Tensors together.
# def serving_input_receiver_fn():
#     """An input receiver that expects a serialized tf.Example."""
#     serialized_tf_example = tf.placeholder(dtype=tf.string,
#                                          shape=[1],
#                                          name='input_example_tensor')
#     receiver_tensors = {'examples': serialized_tf_example}
#     features = tf.parse_example(serialized_tf_example, feature_spec)
#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def serving_input_receiver_fn(): # it is working
    receiver_tensors = {
        'Time': tf.placeholder(tf.float32, [None, 1]),
        'V1': tf.placeholder(tf.float32, [None, 1]),
        'V2': tf.placeholder(tf.float32, [None, 1]),
        'V3': tf.placeholder(tf.float32, [None, 1]),
        'V4': tf.placeholder(tf.float32, [None, 1]),
        'V5': tf.placeholder(tf.float32, [None, 1]),
        'V6': tf.placeholder(tf.float32, [None, 1]),
        'V7': tf.placeholder(tf.float32, [None, 1]),
        'V8': tf.placeholder(tf.float32, [None, 1]),
        'V9': tf.placeholder(tf.float32, [None, 1]),
        'V10': tf.placeholder(tf.float32, [None, 1]),
        'V11': tf.placeholder(tf.float32, [None, 1]),
        'V12': tf.placeholder(tf.float32, [None, 1]),
        'V13': tf.placeholder(tf.float32, [None, 1]),
        'V14': tf.placeholder(tf.float32, [None, 1]),
        'V15': tf.placeholder(tf.float32, [None, 1]),
        'V16': tf.placeholder(tf.float32, [None, 1]),
        'V17': tf.placeholder(tf.float32, [None, 1]),
        'V18': tf.placeholder(tf.float32, [None, 1]),
        'V19': tf.placeholder(tf.float32, [None, 1]),
        'V20': tf.placeholder(tf.float32, [None, 1]),
        'V21': tf.placeholder(tf.float32, [None, 1]),
        'V22': tf.placeholder(tf.float32, [None, 1]),
        'V23': tf.placeholder(tf.float32, [None, 1]),
        'V24': tf.placeholder(tf.float32, [None, 1]),
        'V25': tf.placeholder(tf.float32, [None, 1]),
        'V26': tf.placeholder(tf.float32, [None, 1]),
        'V27': tf.placeholder(tf.float32, [None, 1]),
        'V28': tf.placeholder(tf.float32, [None, 1]),
        'Amount': tf.placeholder(tf.float32, [None, 1])
    }

    #Convert give inputs to adjust to the model.
    features = {
        'Time': receiver_tensors['V1'],
        'V1': receiver_tensors['V1'],
        'V2': receiver_tensors['V2'],
        'V3': receiver_tensors['V1'],
        'V4': receiver_tensors['V2'],
        'V5': receiver_tensors['V1'],
        'V6': receiver_tensors['V2'],
        'V7': receiver_tensors['V1'],
        'V8': receiver_tensors['V2'],
        'V9': receiver_tensors['V1'],
        'V10': receiver_tensors['V2'],
        'V11': receiver_tensors['V1'],
        'V12': receiver_tensors['V2'],
        'V13': receiver_tensors['V1'],
        'V14': receiver_tensors['V2'],
        'V15': receiver_tensors['V1'],
        'V16': receiver_tensors['V2'],
        'V17': receiver_tensors['V1'],
        'V18': receiver_tensors['V2'],
        'V19': receiver_tensors['V1'],
        'V20': receiver_tensors['V2'],
        'V21': receiver_tensors['V1'],
        'V22': receiver_tensors['V2'],
        'V23': receiver_tensors['V1'],
        'V24': receiver_tensors['V2'],
        'V25': receiver_tensors['V1'],
        'V26': receiver_tensors['V2'],
        'V27': receiver_tensors['V1'],
        'V28': receiver_tensors['V2'],
        'Amount': receiver_tensors['V1']
    }
    return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=receiver_tensors)


def train(model_dir, data_dir, train_steps):
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data(data_dir)

    # Feature columns describe how to use the input.
    
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn= model_fn,
        model_dir=model_dir,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 2,
        })
    
    # Train the Model.
    batch_size = 100
    classifier.train(input_fn=lambda:train_input_fn(train_x, train_y, batch_size), steps=500)
    metrics = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))
    print(metrics)
#     estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

#     temp_input_fn = functools.partial(train_input_fn, data_dir)

#     train_spec = tf.estimator.TrainSpec(temp_input_fn, max_steps=train_steps)

#     exporter = tf.estimator.LatestExporter('Servo', serving_input_receiver_fn=serving_input_fn)
#     temp_eval_fn = functools.partial(eval_input_fn, data_dir)
#     eval_spec = tf.estimator.EvalSpec(temp_eval_fn, steps=1, exporters=exporter)

#     tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


    print(model_dir)
    # Save the model
    classifier.export_savedmodel(os.path.join('/opt/ml/model','export','Servo'),serving_input_receiver_fn=serving_input_receiver_fn)


def load_data(data_dir, y_name='Class'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    #train_path, test_path = maybe_download()
    with tf.device('/cpu:0'):
        train_path = os.path.join(data_dir, 'creditcard.csv')
    train = pd.read_csv(train_path)
    train_x, train_y = train, train.pop(y_name)
    
    with tf.device('/cpu:0'):
        test_path = os.path.join(data_dir, 'creditcard.csv')
    test = pd.read_csv(test_path)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

    
def main(model_dir, data_dir, train_steps):
    
    
    tf.logging.set_verbosity(tf.logging.INFO)
    train(model_dir, data_dir, train_steps)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        '--data-dir',
        default='/opt/ml/input/data/training',
        type=str,
        help='The directory where the CIFAR-10 input data is stored. Default: /opt/ml/input/data/training. This '
             'directory corresponds to the SageMaker channel named \'training\', which was specified when creating '
             'our training job on SageMaker')

    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        '--model-dir',
        default='/opt/ml/model',
        type=str,
        help='The directory where the model will be stored. Default: /opt/ml/model. This directory should contain all '
             'final model artifacts as Amazon SageMaker copies all data within this directory as a single object in '
             'compressed tar format.')

    args_parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.')
    args = args_parser.parse_args()
    main(**vars(args))
