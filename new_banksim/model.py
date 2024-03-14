import math
from datetime import timedelta, datetime
from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import time
import os
# from numba import jit

from activation import MultiStepActivation
from agents.bank import Bank
from agents.central_bank import CentralBank
from agents.clearing_house import ClearingHouse, RealSectorClearingHouse
from agents.corporate_client import CorporateClient
from agents.depositor import Depositor
from exogeneous_factors import ExogenousFactors, SimulationType, InterbankPriority
from strategies.bank_ewa_strategy import BankEWAStrategy
from util import Util


class BankingModel(Model):
    """
    BankSim is a banking agent-based simulation framework developed in Python 3+.

    Its main goal is to provide an out-of-the-box simulation tool to study the impacts of a broad range of regulation policies over the banking system.

    The basic model is based on the paper by Barroso, R. V. et al., Interbank network and regulation policies: an analysis through agent-based simulations with adaptive learning, published in the Journal Of Network Theory In Finance, v. 2, n. 4, p. 53–86, 2016.

    The paper is available online at https://mpra.ub.uni-muenchen.de/73308.
    """

    def __init__(self, simulation_type='HighSpread', exogenous_factors=None, number_of_banks=None):
        super().__init__()
        # Simulation data
        self.simulation_type = SimulationType[simulation_type]
        self.current_step = 0
        BankingModel.update_exogeneous_factors_by_simulation_type(self.simulation_type)

        BankingModel.update_exogeneous_factors(exogenous_factors, number_of_banks)

        # Economy data
        self.numberBanks = ExogenousFactors.numberBanks
        self.numberFirms = ExogenousFactors.numberCorporateClientsPerBank * ExogenousFactors.numberBanks
        self.numberHighRiskFirms = ExogenousFactors.numberHighRiskCorporateClientsPerBank * ExogenousFactors.numberBanks
        self.depositInterestRate = ExogenousFactors.depositInterestRate
        self.interbankInterestRate = ExogenousFactors.interbankInterestRate
        self.liquidAssetsInterestRate = ExogenousFactors.liquidAssetsInterestRate
        self.interbankLendingMarketAvailable = ExogenousFactors.interbankLendingMarketAvailable

        # Bank strategy data
        self.bankStrategies = BankEWAStrategy.bank_ewa_strategy_list()
        self.bankStrategiesId = np.array(range(len(self.bankStrategies)))
        self.bankStrategiesLength = len(self.bankStrategies)

        # Scheduler
        self.schedule = MultiStepActivation(self)

        # Central Bank
        _params = (ExogenousFactors.centralBankLendingInterestRate,
                   ExogenousFactors.offersDiscountWindowLending,
                   ExogenousFactors.minimumCapitalAdequacyRatio,
                   not ExogenousFactors.isCentralBankZeroIntelligenceAgent,
                   ExogenousFactors.DefaultEWADampingFactor)
        self.schedule.add_central_bank(CentralBank(*_params, self))

        # Clearing House
        _params = (self.numberBanks,
                   ExogenousFactors.isClearingGuaranteeAvailable)
        self.schedule.add_clearing_house(ClearingHouse(*_params, self))

        # Real sector Clearing House
        _params = (self.numberBanks, self.numberFirms, self.numberHighRiskFirms)

        self.schedule.add_real_clearing_house(RealSectorClearingHouse(*_params, self))

        # Banks
        _params = (ExogenousFactors.bankSizeDistribution,
                   not ExogenousFactors.areBanksZeroIntelligenceAgents,
                   ExogenousFactors.DefaultEWADampingFactor)
        for _ in range(self.numberBanks):
            bank = Bank(*_params, self, _)
            self.schedule.add_bank(bank)
        self.normalize_banks()

        _params_depositors = (
            not ExogenousFactors.areDepositorsZeroIntelligenceAgents,
            ExogenousFactors.DefaultEWADampingFactor)

        # Depositors and Corporate Clients (Firms)
        if ExogenousFactors.standardCorporateClients:
            _params_corporate_clients = [ExogenousFactors.standardCorporateClientDefaultRate,
                                         ExogenousFactors.standardCorporateClientLossGivenDefault,
                                         ExogenousFactors.standardCorporateClientLoanInterestRate]
            _params_high_risk_corporate_clients = [ExogenousFactors.highRiskCorporateClientDefaultRate,
                                                   ExogenousFactors.highRiskCorporateClientLossGivenDefault,
                                                   ExogenousFactors.highRiskCorporateClientLoanInterestRate]
        else:
            _params_corporate_clients = [ExogenousFactors.wholesaleCorporateClientDefaultRate,
                                         ExogenousFactors.wholesaleCorporateClientLossGivenDefault,
                                         ExogenousFactors.wholesaleCorporateClientLoanInterestRate]
            _params_high_risk_corporate_clients = [ExogenousFactors.highRiskCorporateClientDefaultRate,
                                                   ExogenousFactors.highRiskCorporateClientLossGivenDefault,
                                                   ExogenousFactors.highRiskCorporateClientLoanInterestRate]

        firm_id = 0
        high_risk_id = 0
        for bank in self.schedule.banks:
            for i in range(ExogenousFactors.numberDepositorsPerBank):
                depositor = Depositor(*_params_depositors, bank, self)
                bank.depositors.append(depositor)
                self.schedule.add_depositor(depositor)
            for i in range(ExogenousFactors.numberCorporateClientsPerBank):
                if ExogenousFactors.heterogeneousRiskDistribution:
                    _params_corporate_clients[0] = Util().get_random_default_probability(
                        ExogenousFactors.betaParamAlpha, ExogenousFactors.betaParamBeta)
                else:
                    _params_corporate_clients[0] = ExogenousFactors.standardCorporateClientDefaultRate
                corporate_client = CorporateClient(*_params_corporate_clients, 'LowRisk', bank, self, firm_id)
                self.schedule.add_corporate_client(corporate_client)
                firm_id += 1
            for i in range(ExogenousFactors.numberHighRiskCorporateClientsPerBank):
                # bank.corporateClients.append(corporate_client)
                corporate_client = CorporateClient(*_params_high_risk_corporate_clients, 'HighRisk', bank, self,
                                                   high_risk_id)
                self.schedule.add_corporate_client(corporate_client)
                high_risk_id += 1

    def step(self):
        self.current_step = 1 #+  0.5*self.current_step +
        self.schedule.reset_cycle()
        self.schedule.period_0()
        self.schedule.period_1()
        self.schedule.period_2()


    def run_model(self, n):
        k = 1000
        start = time.perf_counter()
        for i in range(n):
            self.step()
            if i % k == 0:
                end = time.perf_counter()
                time_counter(start, end, n, i)

        self.running = False

    def normalize_banks(self):
        # Normalize banks size and Compute market share (in % of total assets)
        total_size = sum([_.initialSize for _ in self.schedule.banks])
        factor = self.numberBanks / total_size
        for bank in self.schedule.banks:
            bank.marketShare = bank.initialSize / total_size
            bank.initialSize *= factor

    @staticmethod
    def update_exogeneous_factors(exogenous_factors, number_of_banks):
        if isinstance(exogenous_factors, dict):
            for key, value in exogenous_factors.items():
                setattr(ExogenousFactors, key, value)

        if number_of_banks:
            ExogenousFactors.numberBanks = number_of_banks

    @staticmethod
    def update_exogeneous_factors_by_simulation_type(simulation_type):
        if simulation_type == SimulationType.HighSpread:
            pass
        if simulation_type == SimulationType.LowSpread:
            ExogenousFactors.standardCorporateClientLoanInterestRate = 0.06
        elif simulation_type == SimulationType.ClearingHouse:
            ExogenousFactors.isClearingGuaranteeAvailable = True
        elif simulation_type == SimulationType.ClearingHouseLowSpread:
            ExogenousFactors.isClearingGuaranteeAvailable = True
            ExogenousFactors.standardCorporateClientLoanInterestRate = 0.06
        elif simulation_type == SimulationType.Basel:
            ExogenousFactors.standardCorporateClients = False
            ExogenousFactors.isCentralBankZeroIntelligenceAgent = False
            ExogenousFactors.isCapitalRequirementActive = True
            ExogenousFactors.interbankPriority = InterbankPriority.RiskSorted
            ExogenousFactors.standardCorporateClientDefaultRate = 0.05
        elif simulation_type == SimulationType.BaselBenchmark:
            ExogenousFactors.standardCorporateClients = False
            ExogenousFactors.standardCorporateClientDefaultRate = 0.05
        elif simulation_type == SimulationType.DepositInsurance:
            ExogenousFactors.areDepositorsZeroIntelligenceAgents = False
            ExogenousFactors.isDepositInsuranceAvailable = True
        elif simulation_type == SimulationType.DepositInsuranceBenchmark:
            ExogenousFactors.areDepositorsZeroIntelligenceAgents = False


class MyModel(BankingModel):
    """
    BankSim is a banking agent-based simulation framework developed in Python 3+.

    Its main goal is to provide an out-of-the-box simulation tool to study the impacts of a broad range of regulation policies over the banking system.

    The basic model is based on the paper by Barroso, R. V. et al., Interbank network and regulation policies: an analysis through agent-based simulations with adaptive learning, published in the Journal Of Network Theory In Finance, v. 2, n. 4, p. 53–86, 2016.

    The paper is available online at https://mpra.ub.uni-muenchen.de/73308.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                'Alpha': alpha_values,
                'Beta': beta_values,
                'Gamma': gamma_values,
                'MuR': MuR_values,
                'MuIB': MuIB_values,
                'Capital': capital,
                'interbank_interest_rate': interbank_interest_rate,
                "Real_Sector_Interest_Rate": real_sector_interest_rate,

                "Insolvencies": number_of_insolvencies,
                'average_risk': average_risk,
                "Contagions": number_of_contagions,
                "Interbank_Loans": interbank_loans,
                "Central_bank_Loans": central_bank_loans,
                "Real_Sector_Loans": real_sector_loans,
                "Total real sector loans": total_loans,
                "High risk loans": high_risk_loans,
                "High risk loan Ratio": high_risk_ratio,
                'liquid_assets': liquid_assets,
                'Early': early_withdrawal,
                'credit_demand_fullfiled': credit_demand_fullfiled,
                'credit_supply_exahausted': credit_supply_exahausted,

                'Profit': profit
                # 'beta_sd': beta_sd,
            },

            agent_reporters={"Beta": lambda a: a.currentlyChosenStrategy.get_beta_value() if a.is_bank else 0,
                             "MuR Value": lambda a: a.currentlyChosenStrategy.get_MuR_value() if a.is_bank else 0,
                             "MuIB Value": lambda a: a.currentlyChosenStrategy.get_MuIB_value() if a.is_bank else 0,
                             "Alpha": lambda a: a.currentlyChosenStrategy.get_alpha_value() if a.is_bank else 0,
                             "Market Interest rate Value": lambda a: a.get_real_sector_interest_rate(
                                 0) if a.is_bank else 0,
                             "Profit": lambda a: a.lastProfit if a.is_bank else 0,
                             'Highest Prob': lambda a: np.max(a.P) if a.is_bank else 0,
                             'Entropy': lambda a:a.probability_entropy if a.is_bank else 0,
                             "Loan Value": lambda a: a.balanceSheet.totalNonFinancialLoans if a.is_bank else 0,

                             'is_bank': 'is_bank'}
        )

    def step(self):
        start = time.perf_counter()
        super().step()
        end = time.perf_counter()
        # print("Elapsed step")
        # DataCollector
        self.datacollector.collect(self)



# Data Collector Functions
def number_of_insolvencies(model):
    return model.schedule.central_bank.insolvencyPerCycleCounter / model.numberBanks


def number_of_contagions(model):
    if model.schedule.central_bank.insolvencyDueToContagionPerCycleCounter > 0:
        print('Contagion',
            model.schedule.central_bank.insolvencyDueToContagionPerCycleCounter / model.schedule.central_bank.insolvencyPerCycleCounter)
        print(model.schedule.central_bank.insolvencyDueToContagionPerCycleCounter)
    return model.schedule.central_bank.insolvencyDueToContagionPerCycleCounter / model.numberBanks


def real_sector_loans(model):
    return np.sum(model.schedule.real_sector_clearing_house.realSectorLendingMatrix)


def capital(model):
    x = []
    for i in model.schedule.banks:
        x.append(-i.balanceSheet.capital)
    return np.mean(x)


def interbank_loans(model):
    x = []
    for i in model.schedule.banks:
        y = i.balanceSheet.interbankLoan
        if y > 0:
            x.append(y)
    return sum(x)


def real_sector_interest_rate(model):
    lending_matrix = model.schedule.real_sector_clearing_house.realSectorLendingMatrix
    interest_matrix = model.schedule.real_sector_clearing_house.realSectorInterestMatrix
    high_risk_lending_matrix = model.schedule.real_sector_clearing_house.highRiskLendingMatrix
    high_risk_interest_matrix = model.schedule.real_sector_clearing_house.highRiskInterestMatrix

    weights_matrix = lending_matrix / np.sum(lending_matrix)
    hr_weights_matrix = high_risk_lending_matrix / np.sum(high_risk_lending_matrix)
    weighted_matrix = np.multiply(weights_matrix, interest_matrix)
    hr_weighted_matrix = np.multiply(hr_weights_matrix, high_risk_interest_matrix)
    hr_weight = np.sum(high_risk_lending_matrix) / (np.sum(lending_matrix) + np.sum(high_risk_lending_matrix))
    std_weight = np.sum(lending_matrix) / (np.sum(lending_matrix) + np.sum(high_risk_lending_matrix))
    average_interest = np.sum(weighted_matrix)
    hr_average_interest = np.sum(hr_weighted_matrix)
    if not np.isnan(np.sum(average_interest * std_weight + hr_average_interest * hr_weight)):
        # print(average_interest * std_weight + hr_average_interest * hr_weight)
        return average_interest * std_weight + hr_average_interest * hr_weight
    else:
        return ExogenousFactors.liquidAssetsInterestRate


def real_sector_spread(model):
    lending_matrix = model.schedule.real_sector_clearing_house.realSectorLendingMatrix
    interest_matrix = model.schedule.real_sector_clearing_house.realSectorInterestMatrix
    weights_matrix = lending_matrix / np.sum(lending_matrix)
    weighted_matrix = np.multiply(weights_matrix, interest_matrix)
    average_interest = np.sum(weighted_matrix)
    return average_interest - ExogenousFactors.depositInterestRate


def interbank_interest_rate(model):
    lending_matrix = np.abs(model.schedule.clearing_house.interbankLendingMatrix) / 2

    interest_matrix = model.schedule.clearing_house.interbankInterestMatrix
    weights_matrix = lending_matrix / np.sum(lending_matrix) if np.sum(
        lending_matrix) != 0 else lending_matrix * np.sum(lending_matrix)
    weighted_matrix = np.multiply(weights_matrix, interest_matrix)
    average_interest = np.sum(weighted_matrix)
    return max(average_interest, ExogenousFactors.liquidAssetsInterestRate)


def unrestricted_interbank_interest_rate(model):
    x = [i.get_interbank_interest_rate() for i in model.schedule.banks]

    # x = [i.currentlyChosenStrategy.get_MuIB_value() for i in model.schedule.banks]
    return np.mean(x)


def liquid_assets(model):
    x = [i.balanceSheet.liquidAssets for i in model.schedule.banks]
    return sum(x)


def total_loans(model):
    x = [i.balanceSheet.totalNonFinancialLoans for i in model.schedule.banks]
    return np.sum(x)


def high_risk_loans(model):
    x = [i.balanceSheet.highRiskLoans for i in model.schedule.banks]
    return np.sum(x)


def high_risk_ratio(model):
    return high_risk_loans(model) / total_loans(model)


def MuR_values(model):
    loan_vector = [i.balanceSheet.totalNonFinancialLoans for i in model.schedule.banks]
    totalLoans = sum(loan_vector)
    weighted = []
    for n, bank in enumerate(model.schedule.banks):
        if totalLoans != 0:
            if loan_vector[n] / totalLoans != 0:
                weighted.append(loan_vector[n] / totalLoans * bank.currentlyChosenStrategy.get_MuR_value())

    # x = [i.currentlyChosenStrategy.get_MuR_value() for i in model.schedule.banks]
    return np.mean(weighted)


def MuIB_values(model):
    x = [i.currentlyChosenStrategy.get_MuIB_value() for i in model.schedule.banks]
    return np.mean(x)


def gamma_values(model):
    x = [i.currentlyChosenStrategy.get_gamma_value() for i in model.schedule.banks]
    return np.mean(x)


def beta_values(model):
    x = [i.currentlyChosenStrategy.get_beta_value() for i in model.schedule.banks]
    return np.mean(x)


def alpha_values(model):
    x = [i.currentlyChosenStrategy.get_alpha_value() for i in model.schedule.banks]
    return np.mean(x)


def beta_sd(model):
    x = [i.currentlyChosenStrategy.get_beta_value() for i in model.schedule.banks]
    return np.std(x)


def profit(model):
    x = [i.lastProfit for i in model.schedule.banks]
    return np.mean(x)


def credit_demand_fullfiled(model):
    x = [i.creditDemandFulfilled for i in model.schedule.corporate_clients][:-1]
    return np.mean(x)


def credit_supply_exahausted(model):
    x = [i.creditSupplyExhausted for i in model.schedule.banks]
    return np.mean(x)


def central_bank_loans(model):
    x = [i.balanceSheet.discountWindowLoan for i in model.schedule.banks]
    return sum(x)


def early_withdrawal(model):
    x = [i.amountEarlyWithdraw for i in model.schedule.depositors]
    return sum(x)


def average_risk(model):
    x = [i.portfolio_risk for i in model.schedule.banks]
    return np.nanmean(x)


def export_model_networks(model, all=False):
    lending_matrix = model.schedule.real_sector_clearing_house.realSectorLendingMatrix
    high_risk_lending_matrix = model.schedule.real_sector_clearing_house.highRiskLendingMatrix
    interbank_matrix = model.schedule.clearing_house.interbankLendingMatrix

    high_risk_interest_matrix = model.schedule.real_sector_clearing_house.highRiskInterestMatrix
    interest_matrix = model.schedule.real_sector_clearing_house.realSectorInterestMatrix


def time_counter(start, end, n, i, print_estimate=True):
    s = end - start
    t = (n - i) * s / (i + 1)
    x = str(timedelta(seconds=t)).split(':')
    print(i)
    print("Elapsed time: " + str(timedelta(seconds=s)))
    # print("Last {} steps took {}s".format(i, round(s, 3)))
    if print_estimate:
        if 3600 < t:
            print("Estimated completion in {}h {}m {}s".format(x[-3], x[-2], x[-1]))
        elif 60 < t:
            print("Estimated completion in {}m {}s".format(x[-2], x[-1]))
        else:
            print("Estimated completion in {}s".format(x[-1]))


def simulate_monetary_experiment(expansive_rate, restrictive_rate, n, iterations):
    if not os.path.isdir(r'simulated_results'):
        path = os.path.join(os.getcwd(), r'simulated_results')
        os.mkdir(path)
    for i in range(iterations):
        for policy in ['Restrictive', 'Expansive']:
            interest = restrictive_rate if policy == "Restrictive" else expansive_rate
            ExogenousFactors.depositInterestRate = ExogenousFactors.liquidAssetsInterestRate = interest
            ExogenousFactors.centralBankLendingInterestRate = interest + 0.06
            ExogenousFactors.betaParamAlpha = 5 if policy == "Restrictive" else 1
            ExogenousFactors.betaParamBeta = 95 if policy == "Restrictive" else 99
            start = time.perf_counter()
            print(policy + " Iteration " + str(i + 1))
            modelo = MyModel()
            modelo.run_model(n)
            Modelo_original = modelo.datacollector.get_model_vars_dataframe()
            filename = r'simulated_results/{}_model'.format(policy) + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'
            Modelo_original.to_csv(
                filename,
                sep=';',
                decimal=',')
            Modelo_original.to_csv(r'output_model.csv')
            end = time.perf_counter()
            time_counter(start, end, 1, n, False)
            print('Saved file: ' + filename)


def simulate_monetary_prudential_experiment_old(expansive_rate, restrictive_rate, car_min, n, iterations):
    if not os.path.isdir(r'simulated_results'):
        path = os.path.join(os.getcwd(), r'simulated_results')
        os.mkdir(path)
    for i in range(iterations):
        for policy in ['Expansive', 'Restrictive']:
            for prudential in ["Prudential", "NoPrudential"]:
                interest = restrictive_rate if policy == "Restrictive" else expansive_rate
                ExogenousFactors.numberBanks = 50
                ExogenousFactors.isCapitalRequirementActive = True if prudential == "Prudential" else False
                ExogenousFactors.minimumCapitalAdequacyRatio = car_min
                ExogenousFactors.depositInterestRate = ExogenousFactors.liquidAssetsInterestRate = interest
                ExogenousFactors.centralBankLendingInterestRate = interest + 0.06
                ExogenousFactors.betaParamAlpha = 5 if policy == "Restrictive" else 1
                ExogenousFactors.betaParamBeta = 95 if policy == "Restrictive" else 99
                ExogenousFactors.standardCorporateClientDefaultRate = ExogenousFactors.betaParamBeta / (
                        ExogenousFactors.betaParamBeta + ExogenousFactors.betaParamAlpha)
                ExogenousFactors.highRiskCorporateClientDefaultRate = 0.15 if policy == "Restrictive" else 0.1
                start = time.perf_counter()
                print(policy + " " + prudential + " Iteration " + str(i + 1))
                modelo = MyModel()
                modelo.run_model(n)
                Modelo_original = modelo.datacollector.get_model_vars_dataframe()
                data_agents = modelo.datacollector.get_agent_vars_dataframe()
                data_networks = {'lending_matrix': modelo.schedule.real_sector_clearing_house.realSectorLendingMatrix,
                                 'high_risk_lending_matrix': modelo.schedule.real_sector_clearing_house.highRiskLendingMatrix,
                                 'interbank_matrix': modelo.schedule.clearing_house.interbankLendingMatrix}

                filename = r'simulated_results/{}_{}_model'.format(policy, prudential) + datetime.now().strftime(
                    "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'

                agents_file = r'simulated_agents/{}_{}_model_agents.csv'.format(policy,
                                                                                prudential) + datetime.now().strftime(
                    "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'

                data_agents.loc[data_agents['is_bank']].to_csv(agents_file, sep=';', decimal=',')
                Modelo_original.to_csv(
                    filename,
                    sep=';',
                    decimal=',')
                Modelo_original.to_csv(r'output_model.csv')
                for network in data_networks:
                    net_filename = r'simulated_networks/{}_{}_{}_model'.format(policy, prudential,
                                                                               network) + datetime.now().strftime(
                        "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'
                    data_networks[network].to_csv(net_filename, sep=';', decimal=',')
                end = time.perf_counter()
                time_counter(start, end, 1, n, False)
                print('Saved file: ' + filename)


def simulate_monetary_prudential_experiment(expansive_rate, restrictive_rate, car_min, n, iterations, developed):
    if not os.path.isdir(r'simulated_results'):
        path = os.path.join(os.getcwd(), r'simulated_results')
        os.mkdir(path)
    for i in range(iterations):
        for policy in ['Restrictive', 'Expansive']:
            for prudential in ["NoPrudential", "Prudential"]:
                interest = restrictive_rate if policy == "Restrictive" else expansive_rate
                ExogenousFactors.numberBanks = 50
                ExogenousFactors.isCapitalRequirementActive = True if prudential == "Prudential" else False
                ExogenousFactors.minimumCapitalAdequacyRatio = car_min
                ExogenousFactors.depositInterestRate = ExogenousFactors.liquidAssetsInterestRate = interest

                if not developed:
                    ExogenousFactors.betaParamAlpha = 10 if policy == "Restrictive" else 1.5
                    ExogenousFactors.betaParamBeta = 90 if policy == "Restrictive" else 98.5
                    ExogenousFactors.standardCorporateClientDefaultRate = ExogenousFactors.betaParamAlpha / (
                            ExogenousFactors.betaParamBeta + ExogenousFactors.betaParamAlpha)
                    ExogenousFactors.highRiskCorporateClientDefaultRate = ExogenousFactors.standardCorporateClientDefaultRate * 2
                else:
                    ExogenousFactors.betaParamAlpha = 5 if policy == "Restrictive" else 0.5
                    ExogenousFactors.betaParamBeta = 95 if policy == "Restrictive" else 99.5

                    ExogenousFactors.standardCorporateClientDefaultRate = ExogenousFactors.betaParamAlpha / (
                            ExogenousFactors.betaParamBeta + ExogenousFactors.betaParamAlpha)
                    ExogenousFactors.highRiskCorporateClientDefaultRate = ExogenousFactors.standardCorporateClientDefaultRate * 2
                ExogenousFactors.centralBankLendingInterestRate = interest + 0.06
                start = time.perf_counter()
                print(policy + " " + prudential + " Iteration " + str(i + 1))
                modelo = MyModel()

                modelo.run_model(n)
                Modelo_original = modelo.datacollector.get_model_vars_dataframe()
                data_agents = modelo.datacollector.get_agent_vars_dataframe()
                dev_string = "Developed" if developed else "Emerging"
                if dev_string == "Developed":
                    filename = r'simulated_results/developed/{}_{}_{}_model'.format(dev_string, policy,
                                                                                    prudential) + datetime.now().strftime(
                        "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'
                else:
                    filename = r'simulated_results/emerging/{}_{}_{}_model'.format(dev_string, policy,
                                                                                   prudential) + datetime.now().strftime(
                        "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'

                agents_file = r'simulated_agents/{}_{}_{}_model_agents'.format(policy, dev_string,
                                                                               prudential) + datetime.now().strftime(
                    "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'
                data_agents.loc[data_agents['is_bank']].to_csv(agents_file, sep=';', decimal=',')

                Modelo_original.to_csv(
                    filename,
                    sep=';',
                    decimal=',')

                Modelo_original.to_csv(r'output_model.csv')
                data_networks = {#'lending_matrix': modelo.schedule.real_sector_clearing_house.realSectorLendingMatrix,
                                 #'high_risk_lending_matrix': modelo.schedule.real_sector_clearing_house.highRiskLendingMatrix,
                                 'interbank_matrix': modelo.schedule.clearing_house.interbankLendingMatrix}
                for network in data_networks:
                    net_filename = r'simulated_networks/{}_{}_{}_{}_model'.format(policy, prudential, dev_string,
                                                                                  network) + datetime.now().strftime(
                        "%Y%m%d-%H%M%S") + '_' + str(n) + '.csv'
                    pd.DataFrame(data_networks[network]).to_csv(net_filename, sep=';', decimal=',')
                end = time.perf_counter()
                time_counter(start, end, 1, n, False)
                print('Saved file: ' + filename)


selection = 1
if __name__ == "__main__":
    if selection:
        for i in range(10):
            list = [False, True] if i % 2 == 0 else [True, False]
            for developed in list:
                if not developed:
                    simulate_monetary_prudential_experiment(0.05, 0.09, 0.08, 15000, 1, developed)
                else:
                    simulate_monetary_prudential_experiment(0.01, 0.03, 0.08, 15000, 1, developed)
    else:
        start = time.perf_counter()
        n = 5000
        modelo = MyModel()
        modelo.run_model(n)

        Modelo_original = modelo.datacollector.get_model_vars_dataframe()
        data_agents = modelo.datacollector.get_agent_vars_dataframe()
        # Modelo_original
        Modelo_original.to_csv(r'output_model.csv', sep=';', decimal=',')
        Modelo_original.to_csv(
            r'simulated_results/restrictive_model' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + str(n) + '.csv',
            sep=';',
            decimal=',')

        data_agents.loc[data_agents['is_bank']].to_csv(r'output_agents.csv', sep=';', decimal=',')
        end = time.perf_counter()
        time_counter(start, end, 1, n, False)

"""

class MyModel(BankingModel):

    BankSim is a banking agent-based simulation framework developed in Python 3+.
    Its main goal is to provide an out-of-the-box simulation tool to study the impacts of a broad range of regulation policies over the banking system.
    The basic model is based on the paper by Barroso, R. V. et al., Interbank network and regulation policies: an analysis through agent-based simulations with adaptive learning, published in the Journal Of Network Theory In Finance, v. 2, n. 4, p. 53–86, 2016.
    The paper is available online at https://mpra.ub.uni-muenchen.de/73308.


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={"Riskier_Clients": ratio_clients,
                             "Low risk": low_risk_clients,
                             "High risk": high_risk_clients,
                             "Low risk loan": low_risk_loan,
                             "High risk loan": high_risk_loan,
                             "Real sector loan": total_real_sector_loans,
                             "Liquid assets": liquid_assets,
                             "Insolvencies": number_of_insolvencies,
                             "Contagions": number_of_contagions,
                             "Interbank_loans": interbank_loans,
                             'Capital': capital,
                             'CAR': car,
                             'CAR mean': car_mean,
                             'CAR CB': car_cb,
                             'Deposits': deposits,
                             'Central Bank loans': central_bank_loans,
                             'Threshold(average)': safety_threshold1,
                             'Threshold(median)': safety_threshold2,
                             'Early': early_withdrawal,
                             'Final': final_withdrawal,
                             'Withdrawals counter': withdrawals_counter,
                             }
        )

    def step(self):
        super().step()
        # DataCollector
        self.datacollector.collect(self)


# Data Collector Functions (all banks)
def number_of_insolvencies(model):
    return model.schedule.central_bank.insolvencyPerCycleCounter / model.numberBanks


def number_of_contagions(model):
    return model.schedule.central_bank.insolvencyDueToContagionPerCycleCounter / model.numberBanks


def total_real_sector_loans(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.balanceSheet.nonFinancialSectorLoanLowRisk + i.balanceSheet.nonFinancialSectorLoanHighRisk)
    return sum(x)


def interbank_loans(model):
    x = []
    for i in model.schedule.banks:
        y = i.balanceSheet.interbankLoan
        if y > 0:
            x.append(y)
    return sum(x)


def ratio_clients(model):
    x = []
    for i in model.schedule.banks:
        z = i.quantityHighRiskcorporateClients / ExogenousFactors.numberCorporateClientsPerBank
        x.append(z)
    return 100 * (statistics.median(x))


def low_risk_clients(model):
    x = []
    for i in model.schedule.banks:
        z = i.quantityLowRiskcorporateClients
        x.append(z)
    return statistics.median(x)


def high_risk_clients(model):
    x = []
    for i in model.schedule.banks:
        z = i.quantityHighRiskcorporateClients
        x.append(z)
    return statistics.median(x)


def low_risk_loan(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.balanceSheet.nonFinancialSectorLoanLowRisk)
    return sum(x)


def high_risk_loan(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.balanceSheet.nonFinancialSectorLoanHighRisk)
    return sum(x)


def liquid_assets(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.balanceSheet.liquidAssets)
    return sum(x)


def capital(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.balanceSheet.capital)
    return -1 * sum(x)


def deposits(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.balanceSheet.deposits)
    return sum(x)


def central_bank_loans(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.balanceSheet.discountWindowLoan)
    return sum(x)


def car(model):
    x = []
    total_risk_weighted_assets = 0
    for i in model.schedule.banks:
        if i.is_solvent():
            total_risk_weighted_assets = i.get_real_sector_risk_weighted_assets()
            if i.is_interbank_creditor():
                total_risk_weighted_assets += i.balanceSheet.interbankLoan
            if total_risk_weighted_assets != 0:
                total_risk_weighted_assets = -i.balanceSheet.capital / total_risk_weighted_assets
        x.append(total_risk_weighted_assets)
    return sum(x)


def car_mean(model):
    x = []
    total_risk_weighted_assets = 0
    for i in model.schedule.banks:
        if i.is_solvent():
            total_risk_weighted_assets = i.get_real_sector_risk_weighted_assets()
            if i.is_interbank_creditor():
                total_risk_weighted_assets += i.balanceSheet.interbankLoan
            if total_risk_weighted_assets != 0:
                total_risk_weighted_assets = -i.balanceSheet.capital / total_risk_weighted_assets
        x.append(total_risk_weighted_assets)
    return statistics.mean(x)


def safety_threshold1(model):
    x = []
    for i in model.schedule.depositors:
        x.append(i.safetyTreshold)
    return statistics.mean(x)


def safety_threshold2(model):
    x = []
    for i in model.schedule.depositors:
        x.append(i.safetyTreshold)
    return statistics.median(x)


def early_withdrawal(model):
    x = []
    for i in model.schedule.depositors:
        x.append(i.amountEarlyWithdraw)
    return sum(x)


def final_withdrawal(model):
    x = []
    for i in model.schedule.depositors:
        x.append(i.amountFinalWithdraw)
    return sum(x)


def withdrawals_counter(model):
    x = []
    for i in model.schedule.banks:
        x.append(i.withdrawalsCounter)
    return sum(x)





"""
