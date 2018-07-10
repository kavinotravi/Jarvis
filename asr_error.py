### Author: Ravi
### This module contains the code to compute ler and wer.

import numpy as np
import pandas as pd

def error_rate(pred, truth):
    """
    This function implements the Levenshtein distance
    to calculate, letter error rate and word error rate.
    https://people.cs.pitt.edu/~kirk/cs1501/Pruhs/Spring2006/assignments/editdistance/Levenshtein%20Distance.htm
    
    Input should be a list.For wer-list of words
    For ler-list of characters, space included
    """
    pred_len = len(pred) + 1
    truth_len = len(truth) + 1
    if(pred_len==1):
        return (truth_len - 1)
    if(truth_len==0):
        return (pred_len - 1)
    
    d = np.zeros((pred_len, truth_len))
    d[0, :] = np.arange(0, truth_len)
    d[:, 0] = np.arange(0, pred_len)
    
    for i in range(1, pred_len):
        for j in range(1, truth_len):
            if(pred[i-1] == truth[j-1]):
                d[i, j] = d[i-1, j-1]
            else:
                sub = d[i-1, j-1] + 1
                ins = d[i, j-1] + 1
                del_ = d[i-1, j] + 1
                d[i, j] = min(sub, ins, del_)
    return int(d[pred_len - 1, truth_len - 1])

def ler(pred, truth):
    """
    Calculates letter error rate using error_rate fn
    pass the string
    """
    pred = pred.strip()
    truth = truth.strip()
    
    pred = pred.lower()
    truth = truth.lower()
    
    list_pred = [i for i in pred]
    list_truth = [i for i in truth]
    ler = error_rate(list_pred, list_truth)/len(list_truth)
    return ler

def wer(pred, truth):
    """
    Calculates Word Error Rate using error_rate fn
    """
    pred = pred.strip()
    truth = truth.strip()
    
    pred = pred.lower()
    truth = truth.lower()
    
    pred_list = pred.split()
    truth_list = truth.split()
    wer = error_rate(pred_list, truth_list)/len(truth_list)
    return wer

def clean_text(str_):
    str_ = str_.strip()
    str_ = str_.split()
    str_out = str_[1:]
    out = ""
    for s in str_out:
        out += " " + s
    out = out.strip()
    return out

def import_data(file):
    df = pd.read_csv(file, header = None)
    df.columns = ["Truth", "Prediction"]
    for i in range(df.shape[0]):
        df.iloc[i, 0] = clean_text(str(df.iloc[i, 0]))
        df.iloc[i, 1] = clean_text(str(df.iloc[i, 1]))
    return df

def compute_er(data):
    """
    This function returns an array of ler and wer
    of the predictions and truth
    """
    ler_ = []
    wer_ = []
    for i in range(data.shape[0]):
        ler_.append(ler(data.iloc[i, 1], data.iloc[i, 0]))
        wer_.append(wer(data.iloc[i, 1], data.iloc[i, 0]))
    ler_ = np.array(ler_)
    wer_ = np.array(wer_)
    
    return ler_, wer_
