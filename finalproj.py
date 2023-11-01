import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data = pd.read_csv('Sales.csv')
print("Description of columns : \n")
print(data.describe())  #decription of each column
print("\nNo.of Null Columns :\n",data.isnull().sum())  #count of null values in columns
data = data.dropna() 
correlation=data.corr(method='pearson')
sns.heatmap(correlation, cmap="coolwarm",annot=True)
plt.show() 
print(data.corr())
x=data[["TV","Radio","Newspaper"]]
y=data["Sales"]
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
model = DecisionTreeRegressor()
model.fit(xtr,ytr)
plt.figure(figsize=(20, 10), dpi=600)  # Set the figure size and DPI for high quality
plot_tree(model, filled=True, feature_names=list(xtr.columns))
plt.savefig("tree_diagram.png", dpi=600)  # Save the diagram as a high-quality PNG image
plt.show()
features = np.array([[200,40,100]])
us=model.predict(features)
print("Advertising cost on TV  : $",features[0][0],)
print("Advertising cost on Radio : $",features[0][1])
print("Advertising cost on Newspaper : $",features[0][2])
print("Predicted Sales value for Units Sold : ",us[0])