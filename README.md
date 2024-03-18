# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dt=pd.read_csv("/content/titanic.csv")
dt
```
![313065463-4e051c84-3ef7-4798-bee6-aca9d4618ab0](https://github.com/dharani18p/EXNO2DS/assets/118343366/3fcbff6a-ba57-4267-ade8-31df58142b0e)


```
dt.info()
```
![313065545-30868b0e-b795-4fc9-9ed8-9ac7a31e4ec7](https://github.com/dharani18p/EXNO2DS/assets/118343366/b788d817-367c-4288-a60c-d9197dd33212)

```
dt.set_index('PassengerId',inplace=True)
dt.describe()
```
![313065603-863b1af9-4fe8-4b3a-9c5b-031231dc4b68](https://github.com/dharani18p/EXNO2DS/assets/118343366/11bbfc01-6b7d-4447-b1bc-7c1d1579a0fa)

```
dt.shape
![313065690-c54585b6-defa-4896-a665-c0aa221303b9](https://github.com/dharani18p/EXNO2DS/assets/118343366/f417f72f-437e-444e-a3a9-f739a02beec9)

```
dt.nunique()
```
![313065765-2e0c3c6c-37fe-4971-aabf-a369ac945730](https://github.com/dharani18p/EXNO2DS/assets/118343366/eb0c3a59-9062-4c78-801c-3b9c010019f8)

```
dt["Survived"].value_counts()
```
![313065900-cfa5f803-1660-41ae-8f53-d6f816849317](https://github.com/dharani18p/EXNO2DS/assets/118343366/c9f09100-c01f-4cf7-a2bd-3326716fbe21)


```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
![313065980-03cdd245-9048-4127-8ca9-c32ef8346eea](https://github.com/dharani18p/EXNO2DS/assets/118343366/a30f263b-9a50-487e-9c76-b7254cb13773)

```
sns.countplot(data=dt,x="Survived")
```
![313066091-a1805f35-d617-4db7-8ac4-68de8ad589e4](https://github.com/dharani18p/EXNO2DS/assets/118343366/48961b47-3dfe-4f4a-9470-a0459a8cef78)

```
dt.Pclass.unique()
```
![313066171-83772692-11f1-4e86-9e6d-4d580daaa813](https://github.com/dharani18p/EXNO2DS/assets/118343366/99ac2641-799a-40e0-adad-a12d938ab831)

```
dt.rename(columns={'Sex':'Gender'},inplace=True)
dt
```
![313066269-e81ad4bb-99b5-4970-8846-c4d5086f8310](https://github.com/dharani18p/EXNO2DS/assets/118343366/f4bb5bd0-b0f4-41ff-bcba-68cf3c4e8593)

```
sns.catplot(x="Gender",col="Survived",kind="count",data=dt,height=5,aspect=.7)
```
![313066359-a079cbbc-4be4-4bce-a753-496226cc776e](https://github.com/dharani18p/EXNO2DS/assets/118343366/b0dd8fff-5fd5-45ce-9724-87c17c6580a2)

```
sns.catplot(x='Survived',hue="Gender",data=dt,kind="count")
```
![313066416-0cf2efd3-cf46-4a7d-b2ca-2b1baa2fdfba](https://github.com/dharani18p/EXNO2DS/assets/118343366/dc78689a-0494-432f-bdcd-14e718cedc6c)


```
dt.boxplot(column="Age",by="Survived")
```
![313066477-bdc5194b-40e1-402a-9f57-2c6281a5850e](https://github.com/dharani18p/EXNO2DS/assets/118343366/3dc8e1ff-5870-441f-92f1-c586315badfd)


```
sns.scatterplot(x=dt["Age"],y=dt["Fare"])
```
![313066552-3c454a46-7da6-4a00-aaad-84a0d722d31e](https://github.com/dharani18p/EXNO2DS/assets/118343366/6605591d-6523-47ee-a46d-4ed6a38bf339)


```
sns.jointplot(x=dt["Age"],y=dt["Fare"],data=dt)
```
![313086049-c126e649-1f51-4a1b-9fe7-560c83881b5c](https://github.com/dharani18p/EXNO2DS/assets/118343366/2ac52678-6c5f-48c7-aa99-8ce4405b33f7)

```
fig,ax1=plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=dt)
```
![313086107-2ad84b3f-3e39-4292-942e-5c9d5ce75b0e](https://github.com/dharani18p/EXNO2DS/assets/118343366/f515397f-6d69-4b6f-abe6-65beebe11c2d)

```
sns.catplot(data=dt,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![313086214-f13e9a90-7141-4b34-8113-ffeb80b41253](https://github.com/dharani18p/EXNO2DS/assets/118343366/d00204ee-9bc5-4f32-b475-82ce141ea48c)

```
corr=dt.corr()
sns.heatmap(corr,annot=True)
```
![313086279-d6145544-4b82-45af-99da-8847f15abd76](https://github.com/dharani18p/EXNO2DS/assets/118343366/07620a02-160a-42d9-ba29-bc503f4e5473)

```
sns.pairplot(dt)
```
![313086352-7014040a-ec3c-4002-84f2-1b9711121b2a](https://github.com/dharani18p/EXNO2DS/assets/118343366/108b27d4-7bb8-4dc4-a299-5e00d48c07f3)

![313086377-355fb56d-74c3-49c0-bf58-147bd4329140](https://github.com/dharani18p/EXNO2DS/assets/118343366/32d293e0-8a65-4f46-b342-9ca528e1ad81)



# RESULT
        Thus,Data Analyzing of the given dataset was successful.
