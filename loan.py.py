#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics


# In[2]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('dataset.csv')


# #### Data Understanding

# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# ###### Here we see that in the dataset we have no null values.

# In[7]:


df.corr()['Creditability']


# ### Exploratory Data Analysis

# In[8]:


df['Creditability'].value_counts() # 1 - Credit-worthy
                                   # 2 - non- Credit worthy


# In[9]:


sns.countplot(x='Creditability',data=df)


# In[10]:


df['Account Balance'].value_counts() # 1 - no running account
                                     # 2 - no balance or debit
                                     # 3 - 0 <= ... < 200 DM
                                     # 4 - ... >= 200 DM or checking account for at least 1 year 


# In[11]:


sns.countplot(x='Account Balance',data=df)


# In[12]:


sns.countplot(x='Account Balance',hue='Creditability',data=df)


# In[13]:


#df['Duration of Credit (month)'].value_counts()


# In[14]:


df['Duration of Credit (month)'].plot.hist()


# In[15]:


df['Payment Status of Previous Credit'].value_counts() # 0 - hesitant payment of previous credits
                                                       # 1 - problematic running account / there are further credits running but at other bank
                                                       # 2 - no previous credits / paid back all previous credits 
                                                       # 3 - no problems with current credits at this bank 
                                                       # 4 - paid back previous credits at this bank


# In[16]:


sns.countplot(x='Payment Status of Previous Credit',data=df)


# In[17]:


sns.countplot(x='Payment Status of Previous Credit',hue='Creditability',data=df)


# In[18]:


df['Purpose'].value_counts() # 0 - Other
                             # 1 - New Car
                             # 2 - Used Car
                             # 3 - Items Of Furniture
                             # 4 - Radio Televison
                             # 5 - household appliances
                             # 6 - repair
                             # 7 - education
                             # 8 - Vacation
                             # 9 - retraining
                             # 10- business


# In[19]:


sns.countplot(x='Purpose',data=df)


# In[20]:


sns.countplot(x='Purpose',hue='Creditability',data=df)


# In[21]:


df['Credit Amount'].value_counts()


# In[22]:


df['Credit Amount'].plot.hist()


# In[23]:


df['Value Savings/Stocks'].value_counts() # 1 - not available / no savings
                                          # 2 - < 100,- DM
                                          # 3 - 100,- <= ... < 500,- DM 
                                          # 4 - 500,- <= ... < 1000,- DM
                                          # 5 - >= 1000,- DM


# In[24]:


sns.countplot(x='Value Savings/Stocks',data=df)


# In[25]:


sns.countplot(x='Value Savings/Stocks',hue='Creditability',data=df)


# In[26]:


df['Length of current employment'].value_counts() # 1- unemployed
                                                  # 2- <= 1 year
                                                  # 3- 1 <= ... < 4 years 
                                                  # 4- 4 <= ... < 7 years
                                                  # 5- >= 7 years 


# In[27]:


sns.countplot(x='Length of current employment',data=df)


# In[28]:


sns.countplot(x='Length of current employment',hue='Creditability',data=df)


# In[29]:


df['Instalment per cent'].value_counts() # 1- >= 35
                                         # 2- 25 <= ... < 35
                                         # 3- 20 <= ... < 25
                                         # 4- < 20


# In[30]:


sns.countplot(x='Instalment per cent',data=df)


# In[31]:


df['Sex & Marital Status'].value_counts() # 1- male: divorced / living apart
                                          # 2- male: single
                                          # 3- male: married / widowed
                                          # 4- female


# In[32]:


sns.countplot(x='Sex & Marital Status',data=df)


# In[33]:


sns.countplot(x='Sex & Marital Status',hue='Creditability',data=df)


# In[34]:


df['Guarantors'].value_counts() # 1- None
                                # 2- Co-Applicant
                                # 3- Gurantor


# In[35]:


sns.countplot(x='Guarantors',data=df)


# In[36]:


sns.countplot(x='Guarantors',hue='Creditability',data=df)


# In[37]:


df['Duration in Current address'].value_counts() # 1 - < 1 year
                                                 # 2 - 1 <= ... < 4 years
                                                 # 3 - 4 <= ... < 7 years
                                                 # 4 - >= 7 years


# In[38]:


sns.countplot(x='Duration in Current address',data=df)


# In[39]:


sns.countplot(x='Duration in Current address',hue='Creditability',data=df)


# In[40]:


df['Most valuable available asset'].value_counts() # 1 - not available / no assets
                                                   # 2 - Car / Other
                                                   # 3 - Savings contract with a building society / Life insurance
                                                   # 4 - Ownership of house or land


# In[41]:


sns.countplot(x='Most valuable available asset',data=df)


# In[42]:


sns.countplot(x='Most valuable available asset',hue='Creditability',data=df)


# In[43]:


df['Age (years)'].plot.hist()


# In[44]:


df['Concurrent Credits'].value_counts() # 1 - at other banks
                                        # 2 - at department store or mail order house  
                                        # 3 - no further running credits


# In[45]:


sns.countplot(x='Concurrent Credits',data=df)


# In[46]:


sns.countplot(x='Concurrent Credits',hue='Creditability',data=df)


# In[47]:


df['Type of apartment'].value_counts()   # 1 - free apartment
                                         # 2 - rented flat
                                         # 3 - owner-occupied flat


# In[48]:


sns.countplot(x='Type of apartment',data=df)


# In[49]:


sns.countplot(x='Type of apartment',hue='Creditability',data=df)


# In[50]:


df.columns


# In[51]:


df['No of Credits at this Bank'].value_counts() # 1 - One
                                                # 2 - Two or Three
                                                # 3 - Four or Five
                                                # 4 - Six or more
             


# In[52]:


sns.countplot(x='No of Credits at this Bank',data=df)


# In[53]:


sns.countplot(x='No of Credits at this Bank',hue='Creditability',data=df)


# In[54]:


df['Occupation'].value_counts() # 1 - unemployed / unskilled with no permanent residence
                                # 2 - unskilled with permanent residence
                                # 3 - skilled worker / skilled employee / minor civil servant
                                # 4 - executive / self-employed / higher civil servant


# In[55]:


sns.countplot(x='Occupation',data=df)


# In[56]:


sns.countplot(x='Occupation',hue='Creditability',data=df)


# In[57]:


df['No of dependents'].value_counts() # 1 - 3 and more 
                                      # 2 - 0 to 2 


# In[58]:


sns.countplot(x='No of dependents',data=df)


# In[59]:


sns.countplot(x='No of dependents',hue='Creditability',data=df)


# In[60]:


df['Telephone'].value_counts() # 1 - no
                               # 2 - yes


# In[61]:


sns.countplot(x='Telephone',data=df)


# In[62]:


sns.countplot(x='Telephone',hue='Creditability',data=df)


# In[63]:


df['Foreign Worker'].value_counts() # 1 - yes
                                    # 2 - No


# In[64]:


sns.countplot(x='Foreign Worker',data=df)


# In[65]:


sns.countplot(x='Foreign Worker',hue='Creditability',data=df)


# ###### We can also plot the graph via loop but readability will decrease.

# In[66]:


df2=df.drop(['Duration of Credit (month)','Credit Amount','Age (years)'],axis=1)


# In[67]:


def count_plot():
    for i in df2.columns:
      print(i,plt.show((sns.countplot(x=i,data=df2))))


# In[68]:


count_plot()


# In[69]:


df.info()


# In[70]:


df.head()


# ###### Data Preprocessing

# In[71]:


scalar = MinMaxScaler(feature_range=(0, 1))


# In[72]:


df['Credit Amount']=df['Credit Amount'].array.reshape(-1,1)


# In[73]:


df[['Credit Amount']]=scalar.fit_transform(df[['Credit Amount']])


# In[74]:


df['Credit Amount']


# In[75]:


df.head()


# In[76]:


X=df.drop(['Creditability'],axis=1)
Y=df['Creditability']


# In[77]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=52)


# In[78]:


model_params={
   'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params':{
            'criterion':['gini','entropy'],
            'max_depth':[1,2,3,4,5,6,7]
        }
   },
    "svm":{
        'model':SVC(),
        'params':{
            'C':[10,20,30],
            'kernel':['rbf','linear']
        }
    },
    'random forest':{
        'model':RandomForestClassifier(),
        'params':{
            "n_estimators":[50,60,70,90,100,110,120,200],
             "criterion":['gini','entropy'],
              'max_depth':[2,3,4,5,6,7,8,9,10]
    
        }
    },
    'knn':{
        'model':KNeighborsClassifier(),
        'params':{
            'n_neighbors':[3,4,5,6,7,8]
        }
    }
    
        }
   


# In[79]:


for model_name, mp in model_params.items():
    print(mp)


# In[80]:


scores = []
for model_name, mp in model_params.items():
    clf1 = GridSearchCV(mp['model'], mp['params'], cv=5,return_train_score=False)
    clf1.fit(X_train,Y_train)
    scores.append({
        'model' : model_name,
        'best_score': clf1.best_score_,
        'best_params': clf1.best_params_
    })


# In[81]:


best_parameter=pd.DataFrame(scores, columns=['model','best_score','best_params'])


# In[82]:


best_parameter


# ##### Here we see that our accuracy of model of unbalanced data

# #### Feature selection of unbalanced data

# In[83]:


plt.figure(figsize=(20,10))
corr = X_train.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()


# In[84]:


# with the following function we can select highly correlated feature
# it will remove the first feature that is correlated with anything other feature
def correlation(dataset, threshold):
    col_corr = set() # sel of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
        return col_corr


# In[85]:


corr_features = correlation(X_train, 0.7)
len(set(corr_features))


# In[86]:


# here is no highly correlated feature 


# In[87]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ##### we remove those featurs whose VIF is greater than 10.

# In[88]:


X_train_new=X_train.drop(['Occupation','Foreign Worker','Type of apartment','Sex & Marital Status','Concurrent Credits','Age (years)','Telephone','No of dependents'],axis=1)


# In[89]:


scores1 = []
for model_name, mp in model_params.items():
    clf2 = GridSearchCV(mp['model'], mp['params'], cv=5,return_train_score=False)
    clf2.fit(X_train_new,Y_train)
    scores1.append({
        'model' : model_name,
        'best_score': clf2.best_score_,
        'best_params': clf2.best_params_
    })


# In[90]:


best_parameter1 =pd.DataFrame(scores1, columns=['model','best_score','best_params'])


# In[91]:


best_parameter1


# ### SMOTE Technique (purpose = Balancing the dataset)

# In[93]:


df=SMOTE('minority')
x,y=df.fit_resample(X,Y)
x=DataFrame(x,columns=X.columns)
y=DataFrame(y,columns=['Creditability'])


# In[94]:


print(x.shape)
print(y.shape)


# ###### Accuracy and best parametre finding on balanced dataset without usinh feature selection technique

# In[95]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=52)


# In[96]:


scores2 = []
for model_name, mp in model_params.items():
    clf3 = GridSearchCV(mp['model'], mp['params'], cv=5,return_train_score=False)
    clf3.fit(x_train,y_train)
    scores2.append({
        'model' : model_name,
        'best_score': clf3.best_score_,
        'best_params': clf3.best_params_
    })


# In[97]:


scores2


# In[98]:


best_parameter3 =pd.DataFrame(scores2, columns=['model','best_score','best_params'])
best_parameter3


# In[99]:


for z in best_parameter3.values:
  print(z)


# ##### Here we are using feature selection method and then check accuracy and fetch best algorithm for our model

# ##### VIP Method

# In[100]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
B = x_train
vif['Features'] = B.columns
vif['VIF'] = [variance_inflation_factor(B.values, i) for i in range(B.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[101]:


x_train_new=x_train.drop(['Occupation','Foreign Worker','Type of apartment','Sex & Marital Status','Concurrent Credits','Age (years)','Telephone','No of dependents','Instalment per cent','Length of current employment'],axis=1)


# In[102]:


scores3 = []
for model_name, mp in model_params.items():
    clf3 = GridSearchCV(mp['model'], mp['params'], cv=5,return_train_score=False)
    clf3.fit(x_train_new,y_train)
    scores3.append({
        'model' : model_name,
        'best_score': clf3.best_score_,
        'best_params': clf3.best_params_
    })


# In[103]:


best_parameter4 =pd.DataFrame(scores3, columns=['model','best_score','best_params'])
best_parameter4


# In[104]:


for z in best_parameter4.values:
  print(z)


# ##### 1. Here we are seeing that after using featrure selecting method (VIP Method) our accuracy of model is decresing so we are not not using feature selecting on balanced dataset.
# 
# ##### 2. And also all of our model balanced dataset without using feature selecting technique is giving best accuracy.
# 
# ##### 3. in this random forest is giving best accuracy approx. 83% accuracy so we are selecting this algorithm to predicting our model.

# ### RANDOM FOREST

# In[105]:


random_forest=RandomForestClassifier(criterion='entropy', max_depth= 10, n_estimators= 120,random_state=98)


# In[106]:


random_forest.fit(x_train,y_train)


# In[107]:


y_pred_random_forest_test=random_forest.predict(x_test) # Checking accuracy on test data


# In[108]:


print("Confusion Matrix: \n",metrics.classification_report(y_test,y_pred_random_forest_test))
print("Accuracy:",metrics.accuracy_score(y_test,y_pred_random_forest_test))
print("Precision:",metrics.precision_score(y_test,y_pred_random_forest_test))
print("recall_sensitivity:",metrics.recall_score(y_test,y_pred_random_forest_test))
print("recall_specificity:",metrics.recall_score(y_test,y_pred_random_forest_test))
print("f1_positive:", metrics.f1_score(y_test,y_pred_random_forest_test))
print("f1_negative:", metrics.f1_score(y_test,y_pred_random_forest_test))
print("Hamming Loss:", metrics.hamming_loss(y_test,y_pred_random_forest_test))


# In[109]:


y_pred_random_forest_train=random_forest.predict(x_train) # check accuracy on train data


# In[125]:


print("Confusion Matrix: \n",metrics.classification_report(y_train,y_pred_random_forest_train))
print("Accuracy:",metrics.accuracy_score(y_train,y_pred_random_forest_train))
print("Precision:",metrics.precision_score(y_train,y_pred_random_forest_train))
print("recall_sensitivity:",metrics.recall_score(y_train,y_pred_random_forest_train))
print("recall_specificity:",metrics.recall_score(y_train,y_pred_random_forest_train))
print("f1_positive:", metrics.f1_score(y_train,y_pred_random_forest_train))
print("f1_negative:", metrics.f1_score(y_train,y_pred_random_forest_train))
print("Hamming Loss:", metrics.hamming_loss(y_train,y_pred_random_forest_train))


# In[139]:


len(x_train.columns)


# In[144]:


feature_importance=pd.DataFrame({
    'random_forest':random_forest.feature_importances_ },index=x_train.columns)
     


# In[145]:


feature_importance


# In[147]:



index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,8))
model_feature=ax.barh(index,feature_importance['random_forest'],0.4,color='purple',label='Random Forest')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()


# In[148]:


import pickle
pickle.dump(random_forest, open('model2.pkl','wb'))

random_forest = pickle.load(open('model2.pkl','rb'))


# In[ ]:




