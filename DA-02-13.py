import scipy.stats as st
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
mean=194
std=11.2
high=225
low=175
z_high=st.norm(mean,std).cdf(high)
z_low=st.norm(mean,std).cdf(low)
print(f'probability={(z_high-z_low)*100:.2f}')

x=np.array([13,16,19,22,23,38,47,56,58,63,65,70,71])
z=(x-np.mean(x))/np.std(x)
print(f'{z}')

scalar=StandardScaler()
scalar.fit(x.reshape(-1,1))
scalar_transform=scalar.transform(x.reshape(-1,1))
print(f'standardized data \n:{np.round(scalar_transform,decimals=2)}')

min_scalar=MinMaxScaler()
min_scalar.fit(x.reshape(-1,1))
min_transform=min_scalar.transform(x.reshape(-1,1))

df=pd.concat([pd.DataFrame(scalar_transform),pd.DataFrame(min_transform)],axis=1)
plt.figure()

plt.plot(scalar_transform)
plt.legend("standardized")
plt.plot(min_transform)
plt.legend("normalized")
plt.xlabel("number of observation")
plt.ylabel("magnitude")

plt.show()




