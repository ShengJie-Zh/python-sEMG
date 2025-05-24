import numpy as np

def segment_signal(sig_, labels_, target_, exclude=False):
    mask = labels_ == target_ if not exclude else labels_ != target_
    chunks_ = []
    changes = np.diff(mask.astype(int))
    start_indices = np.where(changes == 1)[0] + 1
    end_indices = np.where(changes == -1)[0] + 1
    if mask.iloc[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if mask.iloc[-1]:
        end_indices = np.append(end_indices, len(mask))
    for start, end in zip(start_indices, end_indices):
        chunk = sig_[start:end]  
        if len(chunk) > 0:
            chunks_.append(chunk)
    while len(chunks_) < 10:
        chunks_.append([])
    return chunks_


