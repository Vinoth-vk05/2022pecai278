import pandas as pd
import numpy as np

class Recommender:
    
    #initialization
    csv_file = pd.DataFrame()
    cols = []
    
    def __init__(self):
        self.csv_file = pd.read_csv("main_dataset.csv", index_col = 0)
        self.cols = list(self.csv_file.columns)

        
    #inserttion
    def insertVal(self):
        shop_name = str(input("Enter the shop name: "))
        distance = int(input("Enter shop location: "))
        category = str(input("Enter the category: "))
        count = int(input("Enter the no.of.products: "))
        
        for i in range(count):
            item, price = str(input("Enter the item<space>price: ")).split(' ')
            self.csv_file.loc[len(self.csv_file.index)] = [shop_name, category, item, price, distance]

ca = Recommender()
ca.insertVal()
print(ca.csv_file)
