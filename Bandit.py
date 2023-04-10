
#ogger = logging.getLogger("MAB Application")
from abc import ABC, abstractmethod
#from logs import *
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#logger.setLevel(logging.DEBUG) # this on you need for you tests.

warnings.filterwarnings('ignore')
np.random.seed(57)


Bandit_Reward = [1, 2, 3, 4]
NUM_TRIALS = 2000
EPS = 0.1
TAU = 1 / 3 
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#ch.setFormatter(CustomFormatter())
#logger.addHandler(ch)
class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
    @abstractmethod
    def __init__(self, p):
        pass
    @abstractmethod
    def __repr__(self):
        pass
    @abstractmethod
    def pull(self):
        pass
    @abstractmethod
    def update(self):
        pass
    @abstractmethod
    def experiment(self):
        pass
    
    def plot1(self, result, reward=True):
        """
        Visualize the performance of each bandit
        Parameters
        ----------
        result: dictionary: results of the experiment from a bandit
        reward: bool: whether to return the cumulative rewards or regrets
        """
        fig, ax = plt.subplots()

        if reward:
            cumulative_sum = np.cumsum(result['rewards'], axis=0)
        else:
            cumulative_sum = np.cumsum(result['regrets'], axis=0)
        for bandit in range(cumulative_sum.shape[1]):
            if reward:
                cum_sum = np.log(cumulative_sum)[:, bandit]
            else:
                cum_sum = cumulative_sum[:, bandit]
            ax.plot(np.arange(cumulative_sum.shape[0]), cum_sum,label=result['bandits'][bandit]
            )

        ax.set_title(
            f"{'Log' if reward else ''} Bandit  {'reward' if reward else 'regret'} comparison")
        plt.legend()
        plt.show()
        
    def comparision(self, eps_greedy_results, thompson_sampling_results, trials=NUM_TRIALS):
        """
        Compare the performances of the two algorithms VISUALLY
        Parameters
        ----------
        eps_greedy_results: dictionary: result of the epsilon greedy experiment
        thompson_results: dictionary: result of the thompson sampling experiment
        trials: int: the number of trials used in the bandit experiments
        """
        fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
        trials = np.arange(0, trials)
        ax[0].plot(trials, np.cumsum(eps_greedy_results['rewards']),label='Epsilon-Greedy')
        ax[0].plot(trials, np.cumsum(thompson_sampling_results['rewards']), label='Thomson sampling')
        ax[0].set_title('Total Rewards')
        ax[0].legend()
        ax[1].plot(trials, np.cumsum(eps_greedy_results['regrets']),label='Epsilon-Greedy')
        ax[1].plot(trials, np.cumsum(thompson_sampling_results['regrets']),label='Thopmson sampling')
        ax[1].set_title('Total Regrets')
        ax[0].legend()
        plt.show()




    def report(self, result_greedy, result_thompson) -> None:
        """
        Report the comparison of the results of 2 bandit experiments
        and save the results as csv
        Parameters
        ----------
        result_greedy: dict: result of the epsilon greedy experiment
        result_thompson: dict: result of the thompson sampling experiment
        """
        print('Epsilon Greedy performance')
        self.plot1(result_greedy)
        self.plot1(result_greedy, reward=False)
        print('average reward is ', np.mean(
            np.sum(result_greedy['rewards'], axis=1)))
        print('average regret is ', np.mean(
            np.sum(result_greedy['regrets'], axis=1)))
        print('---------------------------------------------')
        print('Thompson sampling performance')
        self.plot1(result_thompson)
        self.plot1(result_thompson, reward=False)
        print('average reward is ', np.mean(
            np.sum(result_thompson['rewards'], axis=1)))
        print('average regret is ', np.mean(
            np.sum(result_greedy['regrets'], axis=1)))

        df = pd.DataFrame({'Bandit': [], 'Reward': [], 'Algorithm': []})
        for algo_index, algorithm in enumerate([result_greedy, result_thompson]):
            for index, bandit in enumerate(algorithm['bandits']):
                data = pd.Series({'Bandit': bandit, 'Reward': np.sum(algorithm['rewards'][:, index]),
                                  'Algorithm': f"{'Epsilon Greedy' if algo_index == 0 else 'Thompson sampling'}"})
                df = df.append(data, ignore_index=True)

        df.to_csv('report.csv', index=False)




class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy class 
    """
    
    def __init__(self, true_mean, epsilon=EPS, tau=TAU):
        self.true_mean = true_mean
        # parameters for mu - prior is N(0,1)
        self.m = 0
        self.m_estimate = 0
        self.tau = tau
        self.N = 0
        self.eps = epsilon

    def __repr__(self):
        return f'Return of {self.true_mean}'

    def pull(self):
        """
        Pull a randomly generated value using the true mean
        Returns
        -------
        float
        """
        return (np.random.randn() / np.sqrt(self.tau)) + self.true_mean

    def update(self, x):
        """
        Update the prior estimate of the true mean
        Parameters
        ----------
        x : float: outcome of the bandits experiment
        """
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N

    def experiment(self, trials=NUM_TRIALS, bandit_rewards=Bandit_Reward):
        """
        Performing an experiment to estimate the mean of a 
        Parameters
        ----------
        trials: int: number of trials to use in the experiment
        bandit_rewards: List of floats: list of true bandit rewards
        Returns
        -------
        total_result: dictionary of the results of the experiments:
            
        """
        bandits = [EpsilonGreedy(m) for m in bandit_rewards]
        rewards = np.zeros((trials, len(bandits)))
        regrets = np.zeros((trials, len(bandits)))

        for i in range(0, trials):
            if np.random.random() < EPS / (i + 1):  
                j = np.random.randint(len(bandits))
            else:
                j = np.argmax([b.m_estimate for b in bandits])

            x = bandits[j].pull()
            bandits[j].update(x)
            rewards[i, j] = x
            regrets[i, j] = np.max([bandit.m for bandit in bandits]) - x

        results = {'rewards': rewards,'regrets': regrets, 'bandits': bandits}

        return results

        



class ThompsonSampling(Bandit):
    """
    Thompson Sampling class
    """

    def __init__(self, true_mean, tau=TAU):
        """
        Thompson Sampling Bandit with its true mean
        Parameters
        ----------
        true_mean: float
        tau: optional float
        """
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = tau
        self.N = 0

    def __repr__(self):
        return f'Return {self.true_mean}'

    def pull(self):
        """
        Pull a randomly generated value using the true mean
        Returns
        -------
        float
        """
        return (np.random.randn() / np.sqrt(self.tau)) + self.true_mean

    def update(self, x):
        """
        Update the prior bayesian estimate of the true mean and variance
        Parameters
        ----------
        x: float
        """
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1

    def sample(self):
        """
        Generate a sample for choosing the bandit to pull from
        Returns
        -------
        float
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def experiment(self, trials = NUM_TRIALS, bandit_rewards = Bandit_Reward):
        """
        Perform experiment for estimating the bandit means
        Parameters
        ----------
        trials: int
        bandit_rewards: List of floats
        Returns
        -------
        total_result: dictionary of the experiments results
        """
        bandits = [ThompsonSampling(m) for m in bandit_rewards]
        rewards = np.zeros((trials, len(bandits)))
        regrets = np.zeros((trials, len(bandits)))

        for i in range(trials):
            j = np.argmax([b.sample() for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)
            rewards[i, j] = x
            regrets[i, j] = np.max([bandit.m for bandit in bandits]) - x

        results = {'rewards': rewards,'regrets': regrets, 'bandits': bandits}
        return results


'''
if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")


'''




