#Import necessary packages
import numpy as np
from numpy.random import default_rng
from sklearn import linear_model
import matplotlib.pyplot as plt

#Define the SLR_slope_simulator class
class SLR_slope_simulator:
    
    #Initialize the class with the relevant parameters
    def __init__(self, beta_0, beta_1, x, sigma, seed):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.x = x
        self.n = len(x)
        self.sigma = sigma
        self.rng = default_rng(seed)
        self.slopes = []
        
    #Generate a sample set of responses using the parameters defined by by the class.   
    def generate_data(self, beta_0, beta_1, x, n, rng):
        y = beta_0 + beta_1*x + rng.standard_normal(n)
        return {"parameters": x, "response": y}
    
    #Estimate a slope using the `sklearn` module. Returns the slope estimate.
    def fit_slope(self, x, y):
        reg = linear_model.LinearRegression()
        fit = reg.fit(x.reshape(-1, 1), y)
        return fit.coef_[0]
    
    #Runs a specified number of simulations and saves the estimates to the slope attribute.
    def run_simulations(self, n_reps):
        estimates = np.zeros(shape = (n_reps, 1)) #generates an array of zeroes, of length n_reps
        for i in range(n_reps):
            y = self.generate_data(self.beta_0, self.beta_1, self.x, self.n, self.rng)['response']
            slope = self.fit_slope(self.x, y)
            estimates[i] = slope
        self.slopes = estimates
        
    #Plots the sampling distribution of the estimated slopes using a histogram
    def plot_sampling_distribution(self):
        #Check if the run_simulations() function has been run.
        if len(self.slopes) > 0:
            plt.hist(self.slopes)
            plt.show()
        else: #Return a printed statement
            print("Please run the 'run_simulations()' function before calling 'plot_sampling_distribution()'.")
    
    #Finds the 1-or-2-tailed probability that the true slope is more extreme than a specified value.
    def find_prob(self, value, sided):
        #Check if the run_simulations() function has been run.
        if len(self.slopes) > 0:
            
            #Calculate relevant p-value
            if sided == "above":
                bool_prob = self.slopes > value
                return(bool_prob.mean())
            elif sided == "below":
                bool_prob = self.slopes < value
                return(bool_prob.mean())
            elif sided == "two-sided":
                bool_prob = abs(self.slopes) > value
                return(bool_prob.mean())
            else: #Return a printed statement
                print("Please specify a different 'sided' argument. Valid options are 'above', 'below', and 'two-sided'.")
            
        else: #Return a printed statement
            print("Please run the 'run_simulations()' function before calling 'find_prob()'.")
            

example = SLR_slope_simulator(beta_0 = 12, beta_1 = 2, x = np.array(list(np.linspace(start = 0, stop = 10, num = 11))*3), sigma = 1, seed = 10)
example.plot_sampling_distribution() #Returns a printed error statement
example.run_simulations(10000)
example.plot_sampling_distribution() #Works as expected
print(example.find_prob(2.1, "two-sided"))
print(example.slopes)
                            