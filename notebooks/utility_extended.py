from neuralpredictors.training import LongCycler
import numpy as np
import torch

def calcLossForDataset(model, dataset, neurons, in_bits=False):
    model.eval()
    with torch.no_grad():
        losses = 0
        samples = 0
        for batch_idx, (data_key, batch) in enumerate(LongCycler(dataset)):
            loss = model.loss(*batch, data_key=data_key, use_avg=False)
            losses += loss.item()
            samples += len(batch[0])

        return losses / samples / neurons if in_bits==False else losses / samples / neurons / np.log(2)