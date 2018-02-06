"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        
        # All required variables
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        # Intialize Q Table by the number of action and number of states
        self.q_table = [[rand.random() for i in range(num_actions)] for j in range(num_states)]

        # Initialize T Table if dyna > 0
        if(self.dyna > 0):
            self.tc_table = [[[0.0 for i in range(num_states)]for i in range(num_actions)] for j in range(num_states)]
            self.t_table = [[[0.00001 for i in range(num_states)]for i in range(num_actions)] for j in range(num_states)]    
            self.r_table = [[0.0 for i in range(num_actions)] for j in range(num_states)]    
        

        # Initialize the current action and state
        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        rightActionRate = rand.random()
        if(rightActionRate >= self.rar):
            action = np.argmax(self.q_table[s])
        else:
            action = rand.randint(0, self.num_actions-1)
            
        self.a = action
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new action
        @returns: The selected action
        """

        # Update q-table 
        self.q_table[self.s][self.a] = (1 - self.alpha)*self.q_table[self.s][self.a] + self.alpha*(r + self.gamma
                                * self.q_table[s_prime][np.argmax(self.q_table[s_prime])])

        # Random select the new action
        rightActionRate = rand.random()
        if(rightActionRate >= self.rar):
            action = np.argmax(self.q_table[s_prime])
        else:
            action = rand.randint(0, self.num_actions-1) 

        # ===============================================Dyna-Q===========================================================================

        if(self.dyna > 0):
            self.tc_table[self.s][self.a][s_prime] = self.tc_table[self.s][self.a][s_prime] + 1
            total = sum(self.tc_table[self.s][self.a])
            for i in range(self.num_states):
                self.t_table[self.s][self.a][i] = 1.0*self.tc_table[self.s][self.a][i] / total
            self.r_table[self.s][self.a] = (1 - self.alpha)*self.r_table[self.s][self.a] + self.alpha*r 

            for i in range(self.dyna):
                dyna_s = rand.randint(0, self.num_states-1)
                dyna_a = rand.randint(0, self.num_actions-1)
                s_rate = rand.random()
                current_rate = 0.0
                dyna_s_prime = 0
                
                for t_index in range(self.num_states):
                    existing_rate = self.t_table[dyna_s][dyna_a][t_index]
                    current_rate = current_rate + existing_rate
                    if(current_rate >= s_rate):
                        dyna_s_prime = t_index
                        break

                dyna_r = self.r_table[dyna_s][dyna_a]
                
                self.q_table[dyna_s][dyna_a] = (1 - self.alpha)*self.q_table[dyna_s][dyna_a] + self.alpha*(dyna_r + self.gamma 
                    * self.q_table[dyna_s_prime][np.argmax(self.q_table[dyna_s_prime])])
                
        # =================================================================================================================================

        
        # update the current q-learner state
        self.s = s_prime
        self.a = action
        self.rar = self.rar * self.radr

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action


if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
