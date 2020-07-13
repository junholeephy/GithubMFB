import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.stats import norm, t
import math
np.set_printoptions(precision=4)

def bocd(data, model, hazard, data_length, Message = np.array([1]), std=1):
    "Return run length posterior using Algorithm 1 in Adams & MacKay 2007"
    # 1. Initialization
    RL_dist = 1
    message = Message
    STD = std
    
    # 2. Observe new data
    x = data
    
    # 3. Evaluate predictive probabilities.
    UPM = model.pred_prob_meanOnly(data_length, x, STD)
    #print("UPM : ", UPM)
    
    # 4. Calculate growth probabilities.
    # given 'N' length of UPM, having 'N' length of growth_probs, change point on current scopre 't' excluded.
    growth_probs = UPM * message * (1 - hazard)
    
    # 5. Calculate changepoint probabilities.
    # only for the change point on current scope 't'
    cp_prob = sum(UPM * message * hazard)
    
    # 6. Calculate evidence 
    # overall probability for current time scope 't'
    new_joint = np.append(cp_prob, growth_probs)
    
    # 7. Determine run length distribution
    RL_dist = new_joint
    evidence = sum(new_joint)
    RL_dist /= evidence
    
    # 8. Update sufficient statistics
    model.update_statistics_meanOnly(data_length, x)
    
    return RL_dist, new_joint

class NormalKnownPrecision:

    def __init__(self, mean0, prec0):
        """Initialize model parameters.
        """        
        self.mean0 = mean0
        self.prec0 = prec0
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([prec0])
        self.num = 0
        
    def pred_prob_meanOnly(self, data_length, x, STD):
        """Compute predictive probabilities.
        """
        #d = lambda x, mu, tau: norm.pdf(x, mu, (math.sqrt(1 / tau) + STD))
        d = lambda x, mu, tau: norm.pdf(x, mu, (math.sqrt(1 / tau + STD*STD) ))
        #return np.array(d(x, self.mean_params, self.prec_params)) # using index corresponding data : bad idea
        #print("{}, Expectd Mu : {}, STD  : {}".format(self.num, self.mean_params[-1], math.sqrt(1.0/self.prec_params[-1])))
        #print("E(mu) : ", self.mean_params)
        
        self.num += 1
                
        return np.array([d(x, self.mean_params[i], self.prec_params[i]) for i in range(data_length)])

    def update_statistics_meanOnly(self, data_length, x):
        """Update sufficient statistics.
        """
        # `offsets` is just a clever way to +1 all the sufficient statistics.
        #print("data : ", x)
        offsets = np.arange(1, data_length + 1)
        new_mean_params = (self.mean_params * offsets + x) / (offsets + 1)
        #new_mean_param = (self.mean_params[-1] * data_length + x) / (data_length + 1)
        new_prec_params = (self.prec_params + 1) * self.prec0
        self.mean_params = np.append([self.mean0], new_mean_params)
        #self.mean_params = np.append(self.mean_params, new_mean_param)
        self.prec_params = np.append([self.prec0], new_prec_params)
        #print("data : ", x)
        #print("self.mean_params : ", self.mean_params)
        #print("new_mean_param : ", new_mean_param)
        #print("self.prec0 : ", self.prec0)
        
def bocd_meanNstd(data_list, model, hazard, Message = np.array([1])):
    '''
    Foundation : This is based on BOCPD paper of Adams & MacKay, 2007
    Application : The function is a BOCPD for both mean & std are unknown case.
    The function returns Run-Length distribution & given data-point's probability
        to be a certain sequence counting from a last Change-Point.
    '''
    # 1. Initialization
    RL_dist = None # Nomination of Run-Length distribution
    message = Message # Iterative message which is updated on each iteration
    
    # 2. Observe new data
    x = data_list[-1] # The last point of data list
    data_length = len(data_list)
    #print("### ", x, data_list, data_length)
    
    # 3. Evaluate predictive probabilities.
    UPM = model.pred_prob(data_list=data_list, x=x) # Calculated UPM predictive (all of the collected data included)
    #print("UPM : ", UPM)
    
    # 4. Calculate growth-probabilities : Partial numerator of Run-Length Posterior, excluding the case Change-Point on current scope 't'
    growth_probs = UPM * message * (1 - hazard) # given 'N' length of UPM, having 'N' length of growth-probs,
    
    # 5. Calculate changepoint probabilities : Partial numerator of Run-Length Posterior, only including the case Change-Point on current scope 't'
    cp_prob = sum(UPM * message * hazard) # The summation required for taking into account all possible scenario from 't-1' sequence
    
    # 6. Calculate numerator & denominator (evidence) of Run-Length posterior
    new_joint = np.append(cp_prob, growth_probs) # Overall probability for current time scope 't'
    evidence = sum(new_joint)                    # Evidence of the RL posterior
    
    # 7. Determine Run-Length distribution
    RL_dist = new_joint
    RL_dist /= evidence # The Run-Length distribution
    
    # 8. Update sufficient statistics : Given exponential family-based likelihood and its conjugate prior,
    #                                   the sufficient statistics are update-able with specific form.
    model.update_statistics(data_length=data_length, x=x) # The last data-point explicitly input as
                                                          # we were not able to found elegant way to update.
    return RL_dist, new_joint


class NormalUnKnownMeanPrecision:

    def __init__(self, mu0, gamma0, alpha0, beta0):
        """Initialize model parameters.
        """
        self.mu0 = mu0
        self.gamma0 = gamma0
        self.alpha0 = alpha0
        self.beta0 = beta0
        
        self.mu_params = np.array([mu0])
        self.gamma_params = np.array([gamma0])
        self.alpha_params = np.array([alpha0])
        self.num = 0
        
    def pred_prob(self, data_list, x):
        """Compute UPM predictive probabilities.
        """
        #x = data_list[-1]; print("data_list :",x, data_list)   
        data_list = data_list[:-1] # The last data-point included manually on the following step.
        data_length = len(data_list);
        beta_list = [self.beta0]
        #if data_length > 1:
        #print("**", self.gamma_params)
        for ind in range(data_length):
            r_ind = data_length - ind - 1
            beta = self.beta0 + 0.5*sum((np.array(data_list[r_ind:]) - np.average(np.array(data_list[r_ind:])))**2) \
            + (ind + 1)*self.gamma_params[0]/(self.gamma_params[0] + (ind + 1)) * 0.5 * \
            (np.average(np.array(data_list[r_ind:])) - self.mu_params[0])**2
            beta_list.append(beta)
            #print("1 :",0.5*sum((np.array(data_list[r_ind:]) - np.average(np.array(data_list[r_ind:])))**2))
            #print("2 :",(ind + 1)*self.gamma_params[ind]/(self.gamma_params[ind] + (ind + 1)) * 0.5)
            #print("3 :",(np.average(np.array(data_list[r_ind:])) - self.mu_params[0])**2)
            #print(data_list[r_ind:])
            #print("* ", (np.array(data_list[r_ind:]) - np.average(np.array(data_list[r_ind:]))))
            #print("** ", sum((np.array(data_list[r_ind:]) - np.average(np.array(data_list[r_ind:])))**2))
        #print(self.num)
        #print(np.array(self.alpha_params))
        #print("beta_list : ", np.array(beta_list))
        #print()
        d = lambda x, df, mu, std : t.pdf(x, df, mu, std) # T distribution for conservative modeling
        d_normal = lambda x, mu, std: norm.pdf(x, mu, std) # Normal distribution for general purpose modeling
        
        ArrayForReturn_t = np.array([d(x, 2*self.alpha_params[i], self.mu_params[i], 
                                     np.sqrt(beta_list[i]*(self.gamma_params[i]+1)/(self.alpha_params[i]*self.gamma_params[i]))) \
                                   for i in range(data_length+1)])
        
        ArrayForReturn_norm = np.array([d_normal(x, self.mu_params[i],
                                                  beta_list[i]*(self.gamma_params[i]+1)/(self.alpha_params[i]*self.gamma_params[i]))\
                                          for i in range(data_length+1)])
        #print("{}, Expectd Mu : {}, Expected STD : {}".format(self.num ,self.mu_params[-1]
        #    ,math.sqrt(1/(self.alpha_params[-1]/beta_list[-1])) ))
        #print("   STE Mu : {}, STE Std {}".format(math.sqrt(beta_list[-1]/(self.gamma_params[-1]*(self.alpha_params[-1]+1))),
        #                                         math.sqrt(beta_list[-1]*math.sqrt(math.sqrt(1.0/self.alpha_params[-1])))))
        #print("E(std) : ", np.sqrt(np.array(beta_list)/self.alpha_params))
        #print("E(mu) : ", self.mu_params)
        #print(np.array(beta_list))
        #print(self.alpha_params)
        self.num += 1
        
        #print()
        return ArrayForReturn_t
    

    def update_statistics(self, data_length, x):
        """Update sufficient statistics.
        """
        # `offsets` is just a clever way to +1 all the sufficient statistics.
        #print("data : ", x)
        offsets = np.arange(1, data_length + 1)
        new_mu_params = (self.mu_params * self.gamma_params + x) / (self.gamma_params + 1)
        new_gamma_params = (self.gamma_params + 1)
        new_alpha_params = (self.alpha_params + 0.5)
        
        self.mu_params = np.append([self.mu0], new_mu_params)
        self.gamma_params = np.append([self.gamma0], new_gamma_params)
        self.alpha_params = np.append([self.alpha0], new_alpha_params)
        
        #print("data : ", x)
        #print("self.mean_params : ", self.mean_params)
        #print("new_mean_param : ", new_mean_param)
        #print("self.prec0 : ", self.prec0)
        
class Batch_SlopeT_SteadyState:
    def __init__(self, BatchData, alpha = 0.05):
        self.BatchData = BatchData
        self.alpha = alpha
    
    def inference(self):
        b0 = 0
        b1 = 0
        sigma_a = 0
        sigma_b1 = 0
        
        sum_t = 0
        sum_zt = 0
        sum_tsquare = 0
        sum_t_minus_taverage_square = 0
        
        DataSize = len(self.BatchData)
        
        for idx in range(DataSize):
            sum_t += (idx+1)
            #print(self.BatchData[idx])
            sum_zt += self.BatchData[idx]
            sum_tsquare += (idx+1)*(idx+1)
        taverage = sum_t / DataSize
        for idx in range(DataSize):
            sum_t_minus_taverage_square += (((idx+1) - taverage) * ((idx+1) - taverage))
            
        for idx in range(DataSize):
            b1 += (idx+1)*self.BatchData[idx]
        b1 -= sum_t * sum_zt / (DataSize)
        b1 /= (sum_tsquare - sum_t*sum_t/DataSize)
        b0 = (sum_zt - b1*sum_t)/DataSize
        
        for idx in range(DataSize):
            sigma_a += (self.BatchData[idx]-b1*(idx+1)-b0)**2
        sigma_a /= (DataSize-2)
        sigma_a = np.sqrt(sigma_a)
        sigma_b1 = sigma_a / np.sqrt(sum_t_minus_taverage_square)
        
        t1 = b1/sigma_b1
        #print(b1,"*")
        
        
        if np.abs(t1) <= t.isf(self.alpha/2, DataSize-2):
            return  0
        else:
            return  b1