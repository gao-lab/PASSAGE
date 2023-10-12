"""PASSAGE runner."""

import glob
import time
import random
from pynvml import *
from tqdm import tqdm, trange
from .preprocess import seed_all
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, clip_grad_norm_
from .model import PASSAGE


def get_free_gpu() -> int:
    r"""
    Get index of GPU with least memory usage

    Ref
    ----------
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    """
    nvmlInit()
    index = 0
    max = 0
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        index = i if info.free > max else index
        max = info.free if info.free > max else max

    return index

def create_batches(contrastive_training_graphs, batch_size):
    """
    Creating batches from the training graph list.
    :return batches: List of lists with batches.
    """
    random.shuffle(contrastive_training_graphs)
    batches = []
    for graph in range(0, len(contrastive_training_graphs), batch_size):
        batches.append(contrastive_training_graphs[graph:graph+batch_size])
    return batches

def process_batch(model, optimizer, batch, gradient_clip_norm, device):
    """
    Forward pass with a batch of data.
    :param batch: Batch of graph pair locations.
    :return loss: Loss on the batch.
    """
    optimizer.zero_grad()
    losses = 0
    loss = 0
    if model.freeze:
        for graph_pair in batch:
            data = torch.load(graph_pair)
            for key in data:
                data[key] = data[key].to(device)
            score_1, score_2 = model(data)
            losses = score_1 - score_2
            losses.backward(retain_graph=True)
            clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            loss = loss + losses.item()
        return loss
    else:
        for graph_pair in batch:
            data = torch.load(graph_pair)
            for key in data:
                data[key] = data[key].to(device)
            losses = model(data)
            losses.backward(retain_graph=True)
            clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            loss = loss + losses.item()
        return loss

def run_PASSAGE(single_graphs:Optional[str]='./dataset/single/',
                train_graphs:Optional[str]='./dataset/train/',
                pretrain_epochs:Optional[int]=40,
                contrastive_epochs:Optional[int]=10,
                GATE_hidden_size_1:Optional[int]=128,
                GATE_hidden_size_2:Optional[int]=16,
                attention_pool_size:Optional[int]=16,
                dropout_rate:Optional[int]=0.3,
                weight_decay:Optional[int]=5*10**-4,
                gradient_clip_norm:Optional[int]=3,
                lr:Optional[int]=0.001,
                batch_contrastive:Optional[int]=32,
                save_model:Optional[str]='./dataset/',
                random_state:Optional[int]=33
                ):
    # devive
    try:
        gpu_index = get_free_gpu()
        print(f"Choose GPU:{gpu_index} as device")
    except:
        print('GPU is not available')
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    # initial enumeration
    seed_all(random_state)
    epochs = pretrain_epochs + contrastive_epochs
    single_graphs = glob.glob(single_graphs + "*.pt")
    contrastive_training_graphs = glob.glob(train_graphs + "*.pt")
    number_of_labels = torch.load(single_graphs[0])['features'].shape[1]

    # setup model
    model = PASSAGE(GATE_hidden_size_1, GATE_hidden_size_2, attention_pool_size,
                    number_of_labels, dropout_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()

    print("\nModel training.\n")
    time1 = time.time()
    epochs = trange(epochs, leave=True, desc="Epoch")
    for epoch in epochs:
        if epoch <= pretrain_epochs:
            batch = single_graphs
            loss_sum = 0
            loss_score = process_batch(model, optimizer, batch, gradient_clip_norm, device)
            loss_sum = loss_sum + loss_score
            loss = loss_sum
            epochs.set_description("Pretraining Epoch (Loss=%g)" % round(loss, 5))
        else:
            model.freeze = True
            for parameter in model.GATE.parameters():
                parameter.requires_grad = False
            batches = create_batches(contrastive_training_graphs, batch_contrastive)
            loss_sum = 0
            main_index = 0
            # for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            for index, batch in enumerate(batches):
                loss_score = process_batch(model, optimizer, batch, gradient_clip_norm, device)
                main_index = main_index + len(batch)
                loss_sum = loss_sum + loss_score * len(batch)
                loss = loss_sum / main_index
                epochs.set_description("Contrastive Epoch (Loss=%g)" % round(loss, 5))
    time2 = time.time()
    print('Training model time: %.2f s' % (time2 - time1))

    torch.save(model, save_model+'PASSAGE_model.pt')
    print(f'The trained model was save at {save_model}'+'PASSAGE_model.pt')
    return model