import numpy as np
import matplotlib.pyplot as plt
#input data

X = np.array([
        [-2,4,-1],
        [4,1,-1],
        [1,6,-1],
        [2,4,-1],
        [6,2,-1],
        ])
# output labels
y= np.array([-1,-1,1,1,1])

#plot graph
for d, sample in enumerate(X):
    if d<2:
        plt.scatter(sample[0],sample[1],s=120,marker='_',linewidths=2)
    else:
        plt.scatter(sample[0],sample[1],s=120,marker='+',linewidths=2)
plt.plot([-2,6],[6,0.5])



#lets perform gradient descent to learn seprating hyperplane
def svm_gd(X,Y):
    #intialize svm w8 vecors wd 0
    w=np.zeros(len(X[0]))
    # learning rate eta = 1
    eta =1
    #how many iterations to train for
    epochs =100000
    # store error
    errors =[]
    #training part,gradient descent part
    for epoch in range(1,epochs):
        error =0
        for i, x in enumerate(X):
            #missclassification
            if (Y[i]*np.dot(X[i],w)) < 1:
                w=w+eta*((X[i] *Y[i]) +(-2 *(1/epoch)*w))
                error =1
            else:
                #correct classifiaction
                w=w+eta * (-2*(1/epoch)*w)
        errors.append(error)
        
    #plot the rate of classification errors
    plt.plot(errors,'|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Missclassified')
    plt.show()
    
    return w



w=svm_gd(X,y)


#plot graph
for d, sample in enumerate(X):
    if d<2:
        plt.scatter(sample[0],sample[1],s=120,marker='_',linewidths=2)
    else:
        plt.scatter(sample[0],sample[1],s=120,marker='+',linewidths=2)
 
    

#add test sample
plt.scatter(2,2,s=120,marker ='_',linewidths=2,color='yellow')
plt.scatter(4,3,s=120,marker ='+',linewidths=2,color='blue')

#print hyperplane calculated
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]
x2x3= np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax=plt.gca()
ax.quiver(X,Y,U,V,scale=1,color='blue')


    
    