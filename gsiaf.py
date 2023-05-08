import numpy as np
from scipy.stats import norm

class HMM: # 2 STATES
    def __init__(self, number_of_states, A, PI, O, params) -> None:
        self.number_of_states = number_of_states    # hidden states = 2 for easy case: folded & unfolded
        self.A = A              # transition probability matrix
        self.PI = PI            # initial state distribution -> uniform
        self.data = O           #  observation sequence -> hp: gaussian distribution        
        self.O = (self.data - self.data.mean(axis=0)) / self.data.std(axis=0)      # normalized data 
        self.params = params # fit parameters of the PDF distribution: it is a matrix (number_of_states, 2)
        self.mean = (self.params[:,0] - self.data.mean(axis=0)) / self.data.std(axis=0) #data & params are normalized
        self.std = self.params[:,1] / self.data.std(axis=0)
        #self._gaussian_vectorized = np.vectorize(self._gaussian)
        #self.epsilon = 0.01
        # self.B = np.zeros(shape=(self.number_of_states, self.O.size)) # observation probability matrix
        # for n in range(self.number_of_states):
        #     self.B[n] = self._gaussian_vectorized(self.c[n], self.mean[n], self.std[n], self.O)

    
    '''
        There are 3 problems in HMM Δ:
        - evaluation problem: how to calculate the probability P(O|Δ) of the observation sequence, indicating how much
            the HMM Δ parameters affects the sequence O;
        - uncovering problem: how to find the sequence of states X ={x_1, x_2, ....., x_T} so that it is more likely to
            produce the observation sequence O;
        - learning problem: how to adjust parameters of Δ such as initial state distribution π, transition probability matrix A
            and observation probability matrix B (-> θ = {μ, σ} for continuous observation HMM) 
            so that the quality of HMM Δ is enhanced.
    '''

    def evaluation_problem(self, print_out=False): #FB algo
        '''
            Forward-backward algorithm:
                1 <= t <= T
        '''
        
        '''
            Forward Variable
        '''
        
        # 1. Initializing step
        b = np.zeros(shape=(self.O.size, self.number_of_states)) # in the future must be corrected to account for num_states > 2
        #b = np.array([ self._gaussian(self.mean[0],self.std[0],self.O) , self._gaussian(self.mean[1],self.std[1],self.O) ])
        b[0] = self._gaussian(self.mean, self.std, self.O[0]) # norm.cdf((self.O[0]+self.epsilon-self.mean)/self.std) - norm.cdf((self.O[0]-self.epsilon-self.mean)/self.std)
        print('b[0]',b[0])
#        print('bcshape', np.shape(b),'\n')
 
        forward_variable = np.zeros(shape=(self.O.size, self.number_of_states))
        forward_variable[0] = b[0]*self.PI #initialize
        
        # 2. Recurrence step
        for t in range(self.O.size-1):
            b[t+1] = self._gaussian(self.mean, self.std, self.O[t+1]) #norm.cdf((self.O[t+1]+self.epsilon-self.mean)/self.std) - norm.cdf((self.O[t+1]-self.epsilon-self.mean)/self.std) 
            forward_variable[t+1] = np.sum((forward_variable[t]*self.A.T).T,axis=0)*b[t+1]

        # 3. Evaluation step:
        probability_O_Delta_forw = np.sum(forward_variable[-1]) # = total prob of getting observable O for x_1:t
        if print_out:
            print(f"La probabilità P(O|Δ) = {probability_O_Delta_forw}, ottenuta con il metodo forward-backward sulla forward-variable")

        '''
            Backward Variable
        '''
        
        # 1. Initializing step
        beta = np.ones(shape=(self.O.size, self.number_of_states)) # In this way the T-th elements are initialized to 1.
        
        # 2. Recurrence step
        for k in range(self.O.size-1, 0, -1):
            beta[k-1] = np.sum(self.A * b[k] * beta[k], axis=1)

                
        # 3. Evaluation step:
        probability_O_Delta_back = np.sum(self.PI*b[0]*beta[0]) # = total prob of getting observable O for x_t+1:T
        if print_out:
            print(f"La probabilità P(O|Δ) = {probability_O_Delta_back}, ottenuta con il metodo forward-backward sulla backward-variable")


        self.alpha = np.nan_to_num(forward_variable.copy(), nan=1e-100)
        self.beta = np.nan_to_num(beta.copy(), nan=1e-100)
        self.b = np.nan_to_num(b.copy(), nan=1e-100)
        #self.alpha = forward_variable.copy()
        #self.beta = beta.copy()
        #self.b = b.copy()

        return self.alpha, self.beta, self.b


    def uncovering_problem(self):
        '''
            Viterbi algorithm
        '''
        delta = np.zeros(shape=(self.O.size, self.number_of_states))    # joint optimal criterion
        q = np.zeros(shape=(self.O.size, self.number_of_states))        # backtracking state
        
        # 1. Initialization step:
        delta[0] = self.b[0]*self.PI #self._gaussian_vectorized(self.mean, self.std, self.O[0])*self.PI
        
        # 2. Recurrence step:
        for t in range(self.O.size-1):
            var = np.zeros(shape=self.number_of_states)
            for j in range(self.number_of_states):
                var[j] = max(delta[t]*self.A[:,j])
            delta[t+1] = var*self.b[t+1] #self._gaussian_vectorized(self.mean, self.std, self.O[t+1])
            q[t+1] = np.argmax(delta[t]*self.A, axis=1) #? the state that maximizes delta is stored in the backtracking state
            #w/ axis=0, q[t+1] stores the transition prob(*delta[t]) of the most probable state 
            #~contrary of the escape probability
            # q[t+1] is the most probable state at t+1 given delta[t], A 

        # 3. State sequence backtracking step:
        X = np.zeros(shape=self.O.size, dtype=int)
        X[-1] = np.argmax(delta[-1])
        for k in range(self.O.size-2, -1, -1):
            X[k] = q[k+1][X[k+1]]
        self.X = X.copy()
        #self.delta = delta.copy()
        return self.X

    
    def learning_problem(self):
        '''
            Expectation Maximization (EM) algortihm === Baum-Welch algorithm
        '''
    
        # E-step:
        xi = np.zeros(shape=(self.O.size, self.number_of_states, self.number_of_states))    # segment posterior
        for t in range(0, self.O.size-1):#?
            xi[t] = (self.alpha[t]*self.A.T).T*self.b[t+1]*self.beta[t+1]
            xi[t] /= np.sum(xi[t])

        gamma = np.zeros(shape=(self.O.size, self.number_of_states))
        gamma = self.alpha*self.beta #self.gamma?
        for t in range(self.O.size):
            gamma[t] /= np.sum(gamma[t])#, axis=1)
        
        # M-step:  
        # New transition probability matrix A 
        new_matrix = np.sum(xi, axis=0)/np.sum(xi, axis=(0,2)) #?
        #rt new_matrix = np.sum(xi[1:self.O.size], axis=0)/np.sum(xi[1:self.O.size], axis=(2,0))

        self.A = new_matrix.copy()

        # New initial state distribution π
        new_PI = np.zeros_like(self.PI)
        new_PI = gamma[0]/np.sum(gamma[0])
        self.PI = new_PI.copy()

        # New observation probability matrix B -> new θ = {μ, σ} for continuous observation HMM
        new_mean = np.zeros_like(self.mean)
        new_std = np.zeros_like(self.std)
        for n in range(self.number_of_states):
            new_mean[n] = np.sum(gamma[:,n]*self.O)/np.sum(gamma[:, n]) # out product?
            new_std[n] = np.sqrt(np.sum(gamma[:,n]*(self.O-new_mean[n])**2)/np.sum(gamma[:, n]))
            #new_std[n] = np.sqrt(np.sum(gamma[:,n]* np.outer((self.O-new_mean[n]),(self.O-new_mean[n])) ) /np.sum(gamma[:, n]))
        self.mean, self.std = new_mean.copy(), new_std.copy()

        return self.A, self.PI, self.mean, self.std


    def iteration(self, MAX_ITERATION = 100, epsilon = 0.0001, show=True):
        # epsilon this could be use after, or by comparing the Probability P(O|Δ) in the evaluation-problem
        way = 0
        while way < MAX_ITERATION:
            
            self.evaluation_problem(print_out=False)
            self.learning_problem()
            print('params @ way',way,' are ',self.params,'\n')

            way += 1
            my_variable = way/MAX_ITERATION*100
            if way%(int(MAX_ITERATION/10)) == 0 and show:
                print(f"Loading {int(my_variable)}%")
                
        self.uncovering_problem() #Viterbi only after training

        # denormalize means and covariances
        self.mean = self.mean * self.data.std() + self.data.mean()
        self.std = self.std * self.data.std()

        # Compute the BIC index
        print(f"The BIC index is: {self.BIC()}")

        return self.mean, self.std, self.X, self.A, self.PI

    
    
    # def _doublegaussian(self, params, x):
    #     # params is a vector of the parameters:
    #     # params = [f_F, sigma_F, w_F, f_U, sigma_U, w_U]
    #     (c1, mu1, sigma1, c2, mu2, sigma2) = params
    #     res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
    #       + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    #     return res

    #def _gaussian(self, mu, sigma, x):
    #    return np.exp(-(x-mu)**2./(2.*sigma**2.))/np.sqrt(2*np.pi*sigma**2)

    def _gaussian(self, mu, sigma, x):
        # return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2.*np.pi*sigma**2)
        return norm.pdf(loc=mu, scale=sigma, x=x)
    
    def _mylog(self, x):
        return 1e-100 if x <= 0 else np.log(x)
    
    def log_likelihood(self):
        # return np.sum(self.gamma*np.log(self.b), axis=(1,0))
        return np.log(np.sum(self.alpha[-1, :])) # last row of alpha corresponds to the probability of being in state i at time t = T
    
    def BIC(self):
        '''
        Computes the Bayesian Information Criterion (BIC) for the HMM model on the given data.
        '''
        L = self.log_likelihood()
        n_params = self.number_of_states ** 2 + self.number_of_states * self.O.size
        
        bic = -2 * L + n_params * np.log(self.O.size)
        
        return bic
    
    def LSE(self, a, b, previous_alpha): # to be applied to alpha, beta
        bepha = np.zeros_like(previous_alpha)
        for j in range(self.number_of_states):
            y = []
            for i in range(self.number_of_states):
                y.append(np.log(previous_alpha[i]) + np.log(a[i,j]))
            bepha[j] = np.max(y) + np.log(b[j])
        return np.exp(bepha)