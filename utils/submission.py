import numpy as np
import torch
import pandas as pd
import os


class Submission:
    def __init__(self):
        self.preds = []
        self.class_names = ['X4_mean', 'X11_mean', 'X18_mean',
                   'X26_mean', 'X50_mean', 'X3112_mean',]
        self.pred_df = pd.read_csv('data/test.csv')
        self.submit_df = pd.read_csv('data/sample_submission.csv')

    def submit(self, model_name, file_name):
        pred_df = self.pred_df[['id']].copy()
        target_cols = [x.replace("_mean","") for x in self.class_names]
        pred_df[target_cols] = self.preds.tolist()

        sub_df = self.submit_df[['id']].copy()
        sub_df = sub_df.merge(pred_df, on='id', how='left')

        folder_name = os.path.join('submission', model_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        sub_df.to_csv(os.path.join(folder_name, file_name), index=False)

