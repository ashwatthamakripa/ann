import numpy as np 

class McCullochPitts:
    def __init__(self,input_size):
        self.weights=np.zeros(input_size)
        self.bias=0
        
    def predict(self,inputs ):
        
        lin_comb=np.dot(self.weights,inputs)+self.bias
        return 1 if lin_comb>=0 else 0
    
inputs=np.array([[0,0],[0,1],[1,0]])
tgrt_output=np.array([0,0,1,0])

neural_network=McCullochPitts(2)   #strictly write same

for input in inputs :
    print(f"inputs:{input}")
    if neural_network.predict(input)==1:
        print("true")
    else:
        print ('false' )