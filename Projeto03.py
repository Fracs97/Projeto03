#!/usr/bin/env python
# coding: utf-8

# ## Anotações:
# ### Verificar se os dados estão balanceados, pois se trata de um problema de classificação
# ### Os dados estão todos codificados, então não há porque fazer uma análise exploratória
# ### Existem muitas variáveis, considerar usar redução de dimensionalidade
# ### Testar os algoritmos com validação cruzada e construir boxplots com os resultados
# ### Testar se feature selection melhora o modelo

# ## 1 = insatisfeito, 0 = satisfeito

# In[28]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[71]:


df = pd.read_csv('train.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[53]:


df.isnull().sum()


# ## A coluna ID é inútil

# In[72]:


df = df.iloc[:,1:]


# In[67]:


df.dtypes


# In[68]:


pd.set_option('max_row',None)


# In[73]:


unicos = pd.DataFrame(df.nunique())


# In[74]:


unicos.loc[unicos[0]==1].index


# ## Algumas colunas só tem um valor único, e por isso podem ser descartadas

# In[75]:


df.drop(list(unicos.loc[unicos[0]==1].index),axis=1,inplace=True)


# In[76]:


unicos.sort_values(0,ascending=False)


# ## A coluna var38 tem 57736 valores únicos, sendo que há 76020 linhas. Avaliar se preciso apagar ela

# In[77]:


df.duplicated().sum()


# ## Removendo duplicatas

# In[78]:


df.drop_duplicates(inplace=True)


# In[13]:


sns.countplot(data=df,x='TARGET');


# ## Extremamente desbalanceado

# ## Posso esperar que haja um desempenho muito melhor na identificação da classe 0 do que na classe 1

# ## Posso tentar um balanceamento por undersampling (Tomek links)

# ## É preciso normalizar os dados antes de aplicar PCA

# In[79]:


x_values = df.iloc[:,0:df.shape[1]].values
y_values = df.iloc[:,df.shape[1]-1].values


# In[80]:


min_max = MinMaxScaler()
min_max.fit(x_values)
x_values_s = min_max.transform(x_values)


# In[81]:


x_values_s


# In[82]:


pca = PCA(n_components=50)
pca.fit(x_values_s)
x_componentes = pca_fit.transform(x_values_s)


# In[83]:


pca_fit.explained_variance_ratio_.sum()


# In[84]:


x_componentes.shape


# ## 50 componentes explicam praticamente toda a informação (99,2%)

# ## Fazendo testes sem balanceamento com validação cruzada

# ## RandomForest, Regressão logística, SVM, NaiveBayes, LDA, KNN, XGBClassifier

# In[89]:


kf = KFold(n_splits=10, shuffle=True)


# In[123]:


modelos = [('RandomForest',RandomForestClassifier()),('KNN',KNeighborsClassifier()),('Reg. L.',LogisticRegression()),          ('SVM',SVC()),('NB',GaussianNB()),('LDA',LinearDiscriminantAnalysis()),('XGBoost',XGBClassifier())]


# In[ ]:


resultados = []
for _,modelo in modelos:
    resultados.append(cross_val_score(estimator=modelo,X=x_componentes,y=y_values,cv=kf,scoring='balanced_accuracy'))


# In[125]:


plt.boxplot(x=resultados,labels=[x[0] for x in modelos]);


# ## Os modelos RandomForest, Regressão Logística, LDA e XGBoost preveram perfeitamente.

# ## Testando os modelos que se saíram melhor individualmente para coletar as acurácias por classe

# In[85]:


x_treino, x_teste, y_treino, y_teste = train_test_split(x_componentes,y_values,test_size=0.3,random_state=2)


# In[86]:


x_treino.shape,x_teste.shape


# In[87]:


randf = RandomForestClassifier()
randf.fit(x_treino,y_treino)


# In[88]:


prev_randf = randf.predict(x_teste)


# In[89]:


print(classification_report(y_teste,prev_randf))


# ## O Random Forest já prevê perfeitamente, não há porque ir em frente
