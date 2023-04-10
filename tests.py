"""
Created on Mon Apr 10 21:15:59 2023

@author: Emmajan
"""
from Bandit import ThompsonSampling,EpsilonGreedy
thompson = ThompsonSampling(5)
epsilon_greedy = EpsilonGreedy(5)



result_thompson = thompson.experiment(2000)
result_epsilon_greedy = epsilon_greedy.experiment(2000)



plot1 = thompson.plot1(result_thompson)
plot2 = epsilon_greedy.plot1(result_epsilon_greedy)
plot3 = thompson.report(result_thompson,result_epsilon_greedy)

plot1
plot2
plot3





