import pandas as pd
import numpy as np

class Recommender:
    
    #initialization
    csv_file: pd.DataFrame
    cols = []
    
    def __init__(self):
        self.csv_file = pd.read_csv("main_dataset.csv", header = None)
        self.cols = list(self.csv_file.iloc[0])[1:]
        print(self.cols)
        self.csv_file = pd.read_csv("main_dataset.csv",header = 0)
        
    #inserttion
    def insertVal():
        pass
        

ca = Recommender()