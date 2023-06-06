# pip install sklearn seaborn plotly matplotlib numpy pandas warnings xboost tqdm

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm_notebook
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
# %matplotlib inline
import os
for dirname, _, filenames in os.walk('cc'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Reading The Data-Set 
data=pd.read_csv('C:/Users/rocks/OneDrive/Desktop/Projects/Water-Quality/water_potability.csv')
data.head()

#  EDA
'''

* ph-> pH of water
* Hardness-> Capacity of water to precipitate soap in mg/L
* Solids-> Total dissolved solids in ppm
* Chloramines-> Amount of Chloramines in ppm
* Sulfate-> Amount of Sulfates dissolved in mg/L
* Conductivity-> Electrical conductivity of water in μS/cm
* Organic_carbon-> Amount of organic carbon in ppm
* Trihalomethanes-> Amount of Trihalomethanes in μg/L
* Turbidity-> Measure of light emiting property of water in NTU (Nephelometric Turbidity Units)
* Potability-> Indicates if water is safe for human consumption

'''

# Describe The Data
data.describe()

# Information Of The Data
data.info()

print('There are {} data points and {} features in the data.'.format(data.shape[0],data.shape[1]))


# Null Values
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

for i in data.columns:
    if data[i].isnull().sum()>0:
        print("There are {} null values in {} column.".format(data[i].isnull().sum(),i))

'''
# Handelling Null Values
'''

# ph
data['ph'].describe()

# Filling The Missing Values By Mean
data['ph_mean']=data['ph'].fillna(data['ph'].mean())

data['ph_mean'].isnull().sum()

# Graphical Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
data['ph'].plot(kind='kde', ax=ax)
data.ph_mean.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()

'''
#### The Distribution Is Not Uniform
'''
# Filling The Data With Random Values
def impute_nan(df,variable):
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample

# Uniform Distribution With Random Initialization
impute_nan(data,"ph")  

# ph_random & ph_mean Graph Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
data['ph'].plot(kind='kde', ax=ax)
data.ph_random.plot(kind='kde', ax=ax, color='green')
data.ph_mean.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()

# Uniform Distribution With Random Initialization
impute_nan(data,"Sulfate")

# Sulfate_random Graphical Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
data['Sulfate'].plot(kind='kde', ax=ax)
data["Sulfate_random"].plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()

# Uniform Distribution With Random Initialization
impute_nan(data,"Trihalomethanes")

# Trihalomethanes Graphical Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
data['Trihalomethanes'].plot(kind='kde', ax=ax)
data.Trihalomethanes_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()



data=data.drop(['ph','Sulfate','Trihalomethanes','ph_mean'],axis=1)
data.isnull().sum()

'''
## Check For Correlation
'''
# Graphical Representation
plt.figure(figsize=(20, 17))
matrix = np.triu(data.corr())
sns.heatmap(data.corr(), annot=True,linewidth=.8, mask=matrix, cmap="rocket",cbar=False);


'''
# There Are No Correlated Columns Presebt In The Data
'''
# Graphical Representation
sns.pairplot(data, hue="Potability", palette="husl");



# Graphical Representation
from tqdm import tqdm

def distributionPlot(data):
    fig = plt.figure(figsize=(20, 20))
    num_columns = len(data.columns)
    num_rows = int(np.ceil(num_columns / 3))
    
    for i in tqdm(range(num_columns)):
        fig.add_subplot(num_rows, 3, i + 1)
        sns.distplot(data.iloc[:, i], color="lightcoral", rug=True)
    
    fig.tight_layout(pad=3)

plot_data = data.drop(['Potability'], axis=1)
distributionPlot(plot_data)


# Hardness
data['Hardness'].describe()

# Distribution Plot Of Hardness Graph 
plt.figure(figsize = (16, 7))
sns.distplot(data['Hardness'])
plt.title('Distribution Plot Of Hardness\n', fontsize =  20)
plt.show()

# Hardness WRT Potability Graph
# basic scatter plot
fig = px.scatter(data,range(data['Hardness'].count()), sorted(data['Hardness']),
                 color=data['Potability'],
                 labels={
                     'x': "Count",
                     'y': "Hardness",
                     'color':'Potability'
                     
                 }, template = 'plotly_dark')
fig.update_layout(title='Hardness WRT Potability')
fig.show()

# Plotly Dark Graph
px.histogram(data_frame = data, x = 'Hardness', nbins = 10, color = 'Potability', marginal = 'box',
             template = 'plotly_dark')

'''
# Solids
'''  
data['Solids'].describe()

# Distribution Plot Of Solids Graph
plt.figure(figsize = (16, 7))
sns.distplot(data['Solids'])
plt.title('Distribution Plot Of Solids\n', fontsize =  20)
plt.show()

# Potability Graph
fig = px.scatter(data, sorted(data["Solids"]), range(data["Solids"].count()), color="Potability", facet_col="Potability", 
                 facet_row="Potability")
fig.show()

# Portability Plotly Dark Graph
px.histogram(data_frame = data, x = 'Solids', nbins = 10, color = 'Potability', marginal = 'box',
             template = 'plotly_dark')

# Hardness WRT Potability Graph
# basic scatter plot
fig = px.scatter(data,range(data['Solids'].count()), sorted(data['Solids']),
                 color=data['Potability'],
                 labels={
                     'x': "Count",
                     'y': "Hardness",
                     'color':'Potability'
                     
                 },
                 color_continuous_scale=px.colors.sequential.tempo,
                 template = 'plotly_dark')
fig.update_layout(title='Hardness WRT Potability')
fig.show()


'''
# Chloramines
'''
data['Chloramines'].describe()

# Distribution Plot Of Chloramines Graph
plt.figure(figsize = (16, 7))
sns.distplot(data['Chloramines'])
plt.title('Distribution Plot Of Chloramines\n', fontsize =  20)
plt.show()

# Chloramines WRT Potability Graph
fig = px.line(x=range(data['Chloramines'].count()), y=sorted(data['Chloramines']),color=data['Potability'], labels={
                     'x': "Count",
                     'y': "Chloramines",
                     'color':'Potability'
                     
                 }, template = 'plotly_dark')
fig.update_layout(title='Chloramines WRT Potability')
fig.show()

# Chloramines Graph
fig = px.box(x = 'Chloramines', data_frame = data, template = 'plotly_dark')
fig.update_layout(title='Chloramines')
fig.show()


'''
# # Conductivity
'''
data["Conductivity"].describe()

# Distribution Plot Of Conductivity Graph
plt.figure(figsize = (16, 7))
sns.distplot(data['Conductivity'])
plt.title('Distribution Plot Of Conductivity\n', fontsize =  20)
plt.show()

# Conductivity WRT Potability Graph
fig = px.bar(data, x=range(data['Conductivity'].count()),
             y=sorted(data['Conductivity']), labels={
                     'x': "Count",
                     'y': "Conductivity",
                     'color':'Potability'
                     
                 },
             color=data['Potability']
             ,template = 'plotly_dark')
fig.update_layout(title='Conductivity WRT Potability')
fig.show() 

# Conductivity Graph
group_labels = ['distplot'] # name of the dataset

fig = ff.create_distplot([data['Conductivity']], group_labels)
fig.show()



'''
# Organic_carbon
'''
data['Organic_carbon'].describe()

# Organic_carbon Graph
group_labels = ['Organic_carbon'] # name of the dataset

fig = ff.create_distplot([data['Organic_carbon']], group_labels)
fig.show()


# Number Of Passengers Per Age Group Graph
dt_5=data[data['Organic_carbon']<5]
dt_5_10=data[(data['Organic_carbon']>5)&(data['Organic_carbon']<10)]
dt_10_15=data[(data['Organic_carbon']>10)&(data['Organic_carbon']<15)]
dt_15_20=data[(data['Organic_carbon']>15)&(data['Organic_carbon']<20)]
dt_20_25=data[(data['Organic_carbon']>20)&(data['Organic_carbon']<25)]
dt_25=data[(data['Organic_carbon']>25)]

x_Age = ['5', '5-10', '10-15', '15-20', '25+']
y_Age = [len(dt_5.values), len(dt_5_10.values), len(dt_10_15.values), len(dt_15_20.values),
     len(dt_25.values)]

px.bar(data_frame = data, x = x_Age, y = y_Age, color = x_Age, template = 'plotly_dark',
       title = 'Number Of Passengers Per Age Group')


# Organic_carbon  Organic_carbon Graph With Potability Hue
sns.catplot(x = 'Organic_carbon', y = 'Organic_carbon', hue = 'Potability', data = data, kind = 'box',
            height = 5, aspect = 2)
plt.show()



'''
# Turbidity
'''
data['Turbidity'].describe()

# Turbidity Graph
group_labels = ['Turbidity'] # name of the dataset

fig = ff.create_distplot([data['Turbidity']], group_labels)
fig.show()

data['turbid_class']=data['Turbidity'].astype(int)
data['turbid_class'].unique()

# Turbidity  turbidity_class Graph
px.scatter(data_frame = data, x = 'Turbidity', y = 'turbid_class', color = 'Potability', template = 'plotly_dark')

data=data.drop(['turbid_class'],axis=1)


'''
# ph_random
'''
data['ph_random'].describe()

# ph_random Graph
group_labels = ['ph_random'] # name of the dataset

fig = ff.create_distplot([data['ph_random']], group_labels)
fig.show()

# ph_random & Portability Graph
px.histogram(data_frame = data, x = 'ph_random', nbins = 10, color = 'Potability', marginal = 'box',
             template = 'plotly_dark')


# --------------------------------
fig = px.scatter(data, sorted(data["ph_random"]), range(data["ph_random"].count()), color="Potability", facet_col="Potability", 
                 facet_row="Potability")
fig.show()


'''
# Sulfate_random
'''
data['Sulfate_random'].describe()

# Sulfate_random Graph
group_labels = ['distplot'] # name of the dataset

fig = ff.create_distplot([data['Sulfate_random']], group_labels)
fig.show()

# Sulfate_random & Sulfate_random With Pottability Hue
sns.catplot(x = 'Sulfate_random', y = 'Sulfate_random', hue = 'Potability', data = data, kind = 'box',
            height = 5, aspect = 2)
plt.show()



'''
# Trihalomethanes_random
'''
data['Trihalomethanes_random'].describe()

# Trihalomethanes_random Graph
group_labels = ['Trihalomethanes_random'] # name of the dataset

# Trihalomethanes_random Ployly Dark Graph
fig = ff.create_distplot([data['Trihalomethanes_random']], group_labels)
fig.show()

fig = px.box(x = 'Trihalomethanes_random', data_frame = data, template = 'plotly_dark')
fig.update_layout(title='Trihalomethanes_random')
fig.show()

# Trihalomethane wrt Potability Graph
fig = px.line(x=range(data['Trihalomethanes_random'].count()), y=sorted(data['Trihalomethanes_random']),color=data['Potability'], labels={
                     'x': "Count",
                     'y': "Trihalomethanes",
                     'color':'Potability'
                     
                 }, template = 'plotly_dark')
fig.update_layout(title='Trihalomethane wrt Potability')
fig.show()



'''
# Potability
'''
data['Potability'].describe()

# Potability Plotly Dark Graph
px.histogram(data_frame = data, x = 'Potability', color = 'Potability', marginal = 'box',
             template = 'plotly_dark')

"""
# Data Preprocessing
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X=data.drop(['Potability'],axis=1)
y=data['Potability']

# Since The Data Is Not In A Uniform Shape, We Scale The Data Using Standard Scalar
scaler = StandardScaler()
x=scaler.fit_transform(X)

# split the data to train and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.85,random_state=42)


print("Training data shape:-{} labels{} ".format(x_train.shape,y_train.shape))
print("Testing data shape:-{} labels{} ".format(x_test.shape,y_test.shape))




"""
# Modeling
"""
# ### Logistic Regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0).fit(x_train, y_train)
log.score(x_test, y_test)

# Confusion Matrix testing data Graph
#  Confusion matrix
from sklearn.metrics import confusion_matrix
# Make Predictions
pred1=log.predict(np.array(x_test))
plt.title("Confusion Matrix testing data")
sns.heatmap(confusion_matrix(y_test,pred1),annot=True,cbar=False)
plt.legend()
plt.show()



# ### K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
# Train the model using the training sets
knn.fit(x_train,y_train)

#Predict Output
predicted= knn.predict(x_test) # 0:Overcast, 2:Mild

# onfusion Matrix testing data Graph
#  Confusion matrix
from sklearn.metrics import confusion_matrix
# Make Predictions
pred1=knn.predict(np.array(x_test))
plt.title("Confusion Matrix testing data")
sns.heatmap(confusion_matrix(y_test,pred1),annot=True,cbar=False)
plt.legend()
plt.show()



# ### SVM
from sklearn import svm
from sklearn.metrics import accuracy_score

svmc = svm.SVC()
svmc.fit(x_train, y_train)

y_pred = svmc.predict(x_test)
print(accuracy_score(y_test,y_pred))

# Confusion Matrix testing data Graph
#  Confusion matrix
from sklearn.metrics import confusion_matrix
# Make Predictions
pred1=svmc.predict(np.array(x_test))
plt.title("Confusion Matrix testing data")
sns.heatmap(confusion_matrix(y_test,pred1),annot=True,cbar=False)
plt.legend()
plt.show()



# ### Decision Tree
from sklearn import tree
from sklearn.metrics import accuracy_score

tre = tree.DecisionTreeClassifier()
tre = tre.fit(x_train, y_train)

y_pred = tre.predict(x_test)
print(accuracy_score(y_test,y_pred))

# Confusion Matrix testing data Graph
#  Confusion matrix
from sklearn.metrics import confusion_matrix
# Make Predictions
pred1=tre.predict(np.array(x_test))
plt.title("Confusion Matrix testing data")
sns.heatmap(confusion_matrix(y_test,pred1),annot=True,cbar=False)
plt.legend()
plt.show()



# ### Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# create the model
model_rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=100)

# fitting the model
model_rf=model_rf.fit(x_train, y_train) 

y_pred = model_rf.predict(x_test)
print(accuracy_score(y_test,y_pred))

# Confusion Matrix testing data Graph
#  Confusion matrix
from sklearn.metrics import confusion_matrix
# Make Predictions
pred1=model_rf.predict(np.array(x_test))
plt.title("Confusion Matrix testing data")
sns.heatmap(confusion_matrix(y_test,pred1),annot=True,cbar=False)
plt.legend()
plt.show()



# ### XG Boost
from xgboost import XGBClassifier
from sklearn.metrics import r2_score

xgb = XGBClassifier(colsample_bylevel= 0.9,
                    colsample_bytree = 0.8, 
                    gamma=0.99,
                    max_depth= 5,
                    min_child_weight= 1,
                    n_estimators= 8,
                    nthread= 5,
                    random_state= 0,
                    )
xgb.fit(x_train,y_train)

print('Accuracy Of XGBoost Classifier On Training Set: {:.2f}'
     .format(xgb.score(x_train, y_train)))
print('Accuracy Of XGBoost Classifier On Test Set: {:.2f}'
     .format(xgb.score(x_test, y_test)))

# Test Confusion Matrix Graph
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
plt.figure(figsize = (15, 8))
sns.set(font_scale=1.4) # for label size
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16},cbar=False, linewidths = 1) # font size
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.savefig('conf_test.png')
plt.show()    



# ### SVM Tuned
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
svc=SVC()
param_grid={'C':[1.2,1.5,2.2,3.5,3.2,4.1],'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'degree':[1,2,4,8,10],'gamma':['scale','auto']}
gridsearch=GridSearchCV(svc,param_grid=param_grid,n_jobs=-1,verbose=4,cv=3)
gridsearch.fit(x_train,y_train)

# Test Confusion Matrix Graph
y_pred=gridsearch.predict(x_test)
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
plt.figure(figsize = (15, 8))
sns.set(font_scale=1.4) # for label size
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16},cbar=False, linewidths = 1) # font size
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.savefig('conf_test.png')
plt.show()