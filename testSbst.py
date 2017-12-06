# Crafting and Testing Adversarial Samples Created by a Substitute Model
# 
# Yang Xiao <xiaoy@vt.edu> 11/28/2017

import numpy as np
from matplotlib import pyplot as plt
import helperFunctions as fun
import whiteBoxAttacks as att
import warnings

warnings.filterwarnings("ignore")

# Adversarial Sample Crafting Parameters
epsilon =  0.8					  # FGS parameter
grad_th = 0.1					      # FGS parameter
coeff = 0.8				  # Opt_L_BFGS parameter
testRange = 10000
testLoc = 1100


# Load MNIST data
mnist = np.load('mnist.npz')
X_test = mnist['teData'][0:testRange]        
Y_test = mnist['teLabels'][0:testRange]
(N, nInp) = np.shape(X_test)


# Load Oracle - The same one used in substitute model training
Oracle = np.load('Oracle_h3.npz')     # 3 hidden layers
Oracle_model = dict([(0,Oracle['W1']),(1,Oracle['b1']),(2,Oracle['W2']),(3,Oracle['b2']),
					 (4,Oracle['W3']),(5,Oracle['b3']),(6,Oracle['W4']),(7,Oracle['b4'])])
Oracle_act = Oracle['act']          # oracle's activation function
Oracle_score = fun.NN_score(fun.NN_pred(X_test, Oracle_model, actfun=Oracle_act),Y_test)
print("Oracle Score: ", Oracle_score)


# Import a substitute model
sbst = np.load('sbst_NN_h1.npz')
sbst_model = dict([(0,sbst['W1']),(1,sbst['b1']),(2,sbst['W2']),(3,sbst['b2'])])
sbst_act = sbst['act']   
sbst_score = fun.NN_score(fun.NN_pred(X_test, sbst_model, actfun=sbst_act),Y_test)
print("Sbst Score: ", sbst_score)       

# Predictions
Pred_Ori_sbst   = fun.NN_pred(X_test, sbst_model,   sbst_act)
Pred_Ori_Oracle = fun.NN_pred(X_test, Oracle_model, Oracle_act)


'''
# Crafting adversarial samples - Goodfellow's fast gradient sign method

print('Adversarial Sample Crafting and Testing - coeff = ', coeff, 'epsilon = ', epsilon)
array09 = np.array([0,1,2,3,4,5,6,7,8,9])
evaAtk_result_sbst = np.zeros(N)    # Evasion attack against sbst for every test sample - {0,1}^N
evaAtk_result_orac = np.zeros(N)	# Evasion attack against orac for every test sample - {0,1}^N
tarAtk_result_sbst = np.zeros(N)	# Targeted attack against sbst for every test sample - [0,1]^N
tarAtk_result_orac = np.zeros(N)	# Targeted attack against sbst for every test sample - [0,1]^N

for i in range(N):
	
	#if np.remainder(i,1000) == 0:
	#	print('i = ',i)

	X_adv = att.FastGradSign_h1(X_test[i], epsilon, grad_th, sbst)
	# X_adv = att.Opt_L_BFGS_h1(X_test[i], sbst, coeff)
	Pred_Adv_sbst   = fun.NN_pred(X_adv,  sbst_model,   sbst['act'])
	Pred_Adv_orac   = fun.NN_pred(X_adv,  Oracle_model, Oracle['act'])

	# Success: at least one of the samples in X_adv are mis-classified by the Oracle
	evaAtk_result_sbst[i] = np.sign(np.size(np.extract(Pred_Adv_sbst!=Y_test[i], Pred_Adv_sbst)))
	evaAtk_result_orac[i] = np.sign(np.size(np.extract(Pred_Adv_orac!=Y_test[i], Pred_Adv_orac)))
	tarAtk_result_sbst[i] = np.size(np.extract(Pred_Adv_sbst==array09, Pred_Adv_sbst)) / 10
	tarAtk_result_orac[i] = np.size(np.extract(Pred_Adv_orac==array09, Pred_Adv_orac)) / 10

	print(i,': ', evaAtk_result_sbst[i], evaAtk_result_orac[i],
				  tarAtk_result_sbst[i], tarAtk_result_orac[i])

# Stat
evaAtk_suc_rate_sbst = sum(evaAtk_result_sbst)/N     # Success rate of evasion attack against the substitute
evaAtk_suc_rate_orac = sum(evaAtk_result_orac)/N     # Success rate of evasion attack against the Oracle
tarAtk_suc_rate_sbst = sum(tarAtk_result_sbst)/N     # Success rate of targeted attack against the substitute
tarAtk_suc_rate_orac = sum(tarAtk_result_orac)/N     # Success rate of targeted attack against the Oracle

print('\nResult - coeff = ',coeff, 'epsilon = ', epsilon)
print('Success rate of evasion attack against the substitute : ', evaAtk_suc_rate_sbst)
print('Success rate of evasion attack against the Oracle     : ', evaAtk_suc_rate_orac)
print('Success rate of targeted attack against the substitute: ', tarAtk_suc_rate_sbst)
print('Success rate of targeted attack against the Oracle    : ', tarAtk_suc_rate_orac)

'''

# Test one sample

att.testOneSample(X_test[testLoc], Y_test[testLoc], epsilon, grad_th, coeff, sbst, sbst_model, Oracle, Oracle_model, Pred_Ori_sbst[testLoc], Pred_Ori_Oracle[testLoc])


