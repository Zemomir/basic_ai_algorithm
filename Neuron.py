import torch
import matplotlib.pyplot as plt
class Neuron :
    def __init__(self,x,w,b,funk):
        self.x = x
        self.w = w
        self.b =b
        self.funk= funk
        self.output= self.forward()
    def forward(self):
        return self.funk(self.x @ self.w +self.b)
    def set_w(self,w2):
        self.w = self.w -w2
        self.output= self.forward()
    def set_b(self,b2):
        self.b = self.b-b2 
        self.output= self.forward()
    def visualization_linear_boundary(self,t=0.5):
        """
        bu fonksiyon neuronun yaptığı fonksiyonu görselleştirmek için var
        t: sınıflandırma eşiğidir. Karar doğrusu w·x + b = t olarak çizilir.
        
        """
        try:
            if self.funk != IdentityFunc :
                raise TypeError
            plt.scatter(self.x[:,0].numpy(),self.x[:,1].numpy())
            x1_line = torch.linspace(-0.5, 1.5, 100)
            x2_line = (t- self.b - self.w[0] * x1_line) / self.w[1]
            plt.plot(x1_line.numpy(), x2_line.numpy())
            plt.show()
        except TypeError:
            print("bu fonksiyon sadece fonksiyonu IdentityFunc olan neuronlarda çalışır")
def IdentityFunc(x):
        return x    

if __name__== "__main__":
    x = torch.tensor([(0.,0.),(0.,1.),(1.,1.),(1.,0.)]) #And problemini ele alırsak
    w = torch.tensor([0.1,0.2])
    b = torch.tensor(0.5)
    y = torch.tensor([0.,0.,1.,0.])
    
    neuron1 = Neuron(x,w,b,IdentityFunc)
    print(f"{neuron1.output} istenilen değerlere ulaşamadık w ve b değerlerinin değişmesi lazım")
    neuron1.visualization_linear_boundary() 
    for _ in range(100):
        lr = torch.tensor(0.2)
        delta_w =x.T@(neuron1.output - y) 
        delta_b = sum((neuron1.output - y))
        neuron1.set_b(delta_b*lr)
        neuron1.set_w(delta_w*lr)
    neuron1.visualization_linear_boundary()
    

    
