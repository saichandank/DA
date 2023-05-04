import numpy as np
from math import log2


x=np.array([[1,1,18],[1,2,25],[1,2,50],[1,3,68],[1,4,75],[1,5,65]])
y=np.array([[29],[25],[21],[18],[15],[15]])
s=np.array([1,3.5,30])

print(1+s@np.linalg.inv(x.T@x)@s.T)



#print((3/4)*((1/3)*log2(1/3)+(2/3)*log2(2/3)))

