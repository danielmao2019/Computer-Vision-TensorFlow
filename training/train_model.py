import tensorflow as tf
import data
import os
import time
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)


def train_loop(dataloader, model, loss, optimizer, metric):
    """
    Args:
        dataloader (data.dataloaders.Dataloader).
        model (tf.keras.Model).
        loss (tf.keras.losses.Loss).
        optimizer (tf.keras.optimizers.Optimizer).
        metric (tf.keras.metrics.Metric).
    """
    if not isinstance(dataloader, data.dataloaders.Dataloader):
        raise TypeError(f"[ERROR] Argument \'dataloader\' should be an instance of data.dataloaders.Dataloader. Got {type(dataloader)}.")
    if not isinstance(model, tf.keras.Model):
        raise TypeError(f"[ERROR] Argument \'model\' should be an instance of tf.keras.Model. Got {type(model)}.")
    tot_error = 0
    tot_score = 0
    for inputs, labels in tqdm(dataloader):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            error = loss(labels, logits)
            score = metric(labels, logits)
        grads = tape.gradient(error, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        tot_error += error
        tot_score += score
    avg_error = tot_error / len(dataloader)
    avg_score = tot_score / len(dataloader)
    return avg_error, avg_score


def train_model(specs):
    """
    spec (dict): The specifications of the training. Should have the following keys:
    {
        tag (str): The tag of the model used for model saving.
        dataloader (data.dataloaders.Dataloader).
        model (tf.keras.Model).
        loss (tf.keras.losses.Loss).
        optimizer (tf.keras.optimizers.Optimizer).
        metric (tf.keras.metrics.Metric).
        epochs (int): The number of epochs for which the model to be trained.
        save_model (bool): Save trained models periodically if set to True and do not save any model if set to False.
        load_model (str): The path under 'saved_models' of the model to load.
    }
    """
    tag = specs['tag']
    dataloader = specs['dataloader']
    if specs['load_model'] is None:
        model = specs['model']
        loss = specs['loss']
        optimizer = specs['optimizer']
        metric = specs['metric']
        model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        start_epoch = 0
    else:
        filepath = os.path.join("saved_models", specs['load_model'])
        model = tf.keras.models.load_model(filepath=filepath, custom_objects=specs['custom_objects'], compile=True)
        loss = model.loss
        optimizer = model.optimizer
        metric = model.metrics[0]
        start_epoch = model.signature_def_map['epoch']
    epochs = specs['epochs']
    end_epoch = start_epoch + epochs
    ####################################################################################################
    device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
    model.trainable = True
    with tf.device(device):
        for cur_epoch in range(start_epoch, end_epoch):
            start_time = time.time()
            error, score = train_loop(
                dataloader=dataloader, model=model,
                loss=loss, optimizer=optimizer, metric=metric,
            )
            logging.info(f"Epoch: {cur_epoch:03d}/{end_epoch}, "
                         f"error={error:.6f}, score={score:.6f}, "
                         f"time={time.time()-start_time:.2f}."
                         )
            if specs['save_model'] and cur_epoch % 5 == 0:
                filepath = os.path.join("saved_models", tag, f"{tag}_epoch{cur_epoch:03d}")
                tf.keras.models.save_model(model, filepath, signatures={
                    'epoch': cur_epoch,
                })
