import MLP_train

x_test = MLP_train.x_test
y_test = MLP_train.y_test

def model1_test():

    model1 = MLP_train.model1()
    # compare accuracy
    score1 = model1.evaluate(x_test, y_test, verbose=0)
    print('Test loss1:', score1[0])
    print('Test accuracy1:', score1[1])
    # Test accuracy1: 0.981238067150116

def model2_test():
    model2 = MLP_train.model2()
    # compare accuracy
    score2 = model2.evaluate(x_test, y_test, verbose = 0)
    print('Test loss2:', score2[0])
    print('Test accuracy2:', score2[1])
    #Test accuracy2: 0.9798095226287842

#model1_test()
model2_test()