
# Import matplotlib, numpy and math 
import matplotlib.pyplot as plt 
import numpy as np 
import math 

## sigmoid plot  
x = np.linspace(-6, 6, 100) 
z = 1/(1 + np.exp(-x)) 
  
plt.rcParams['font.size'] = '16'
plt.plot(x, z, linewidth=2) 
plt.xlabel("z", fontsize=18) 
plt.ylabel("Sigmoid(z)", fontsize=18) 
  
plt.show() 


z = 1/(1 + np.exp(-1000)) 
print(z)


## relu plot  
x = np.linspace(-6, 6, 100) 
z = np.maximum(0,x) 
  
plt.rcParams['font.size'] = '20'
plt.plot(x, z, linewidth=2) 
plt.xticks([-6,-4,-2,0,2,4,6])#size='small')
plt.yticks([0,1,2,3,4,5,6])
plt.xlabel("z", fontsize=22) 
plt.ylabel("ReLU(z)", fontsize=22) 
  
plt.show() 


## tanh plot  
x = np.linspace(-6, 6, 100) 
z = 2* (1/(1 + np.exp(-x))) -1 
  
plt.rcParams['font.size'] = '20'
plt.plot(x, z, linewidth=2) 
plt.xticks([-6,-4,-2,0,2,4,6])#size='small')
plt.yticks([-1,-0.5,0,0.5,1])
plt.xlabel("z", fontsize=22) 
plt.ylabel("tanh(z)", fontsize=22) 
  
plt.show() 


## loss function - probability
x = np.linspace(0, 1, 100) 
z = np.log(1-x) * (-1) # class 0

plt.rcParams['font.size'] = '20'
plt.axis( [0, 1, 0, 5] ) # makes sure y ends at 0
plt.plot(x, z, color="blue", linewidth=2, linestyle="-", label=r'Loss if $y=0$') 
# Legende einblenden
plt.legend(loc='upper left', frameon=True, prop={"size":20})
plt.xlabel(r'Predicted probability - $\phi(z)$', fontsize=22) 
plt.ylabel("Loss", fontsize=22) 
plt.show() 

## loss function - probability
x = np.linspace(0, 1, 100) 
z = np.log(x) * (-1) #class 1

plt.rcParams['font.size'] = '20'
plt.axis( [0, 1, 0, 5] ) # makes sure y ends at 0
plt.plot(x, z, color="orange", linewidth=2, linestyle="-", label=r'Loss if $y=1$') 
# Legende einblenden
plt.legend(loc='upper left', frameon=True, prop={"size":20})
plt.xlabel(r'Predicted probability - $\phi(z)$', fontsize=22) 
plt.ylabel("Loss", fontsize=22) 
plt.show() 
