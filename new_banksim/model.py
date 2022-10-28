from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np
import time
# from numba import jit

from activation import MultiStepActivation
from agents.bank import Bank
from agents.central_bank import CentralBank
from agents.clearing_house import ClearingHouse, RealSectorClearingHouse
from agents.corporate_client import CorporateClient
from agents.depositor import Depositor
from exogeneous_factors import ExogenousFactors, SimulationType, InterbankPriority


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
        BankingModel.update_exogeneous_factors_by_simulation_type(self.simulation_type)

        BankingModel.update_exogeneous_factors(exogenous_factors, number_of_banks)

        # Economy data
        self.numberBanks = ExogenousFactors.numberBanks
        self.numberFirms = ExogenousFactors.numberCorporateClientsPerBank * ExogenousFactors.numberBanks
        self.depositInterestRate = ExogenousFactors.depositInterestRate
        self.interbankInterestRate = ExogenousFactors.interbankInterestRate
        self.liquidAssetsInterestRate = ExogenousFactors.liquidAssetsInterestRate
        self.interbankLendingMarketAvailable = ExogenousFactors.interbankLendingMarketAvailable

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
        _params = (self.numberBanks, self.numberFirms)
        self.schedule.add_real_clearing_house(RealSectorClearingHouse(*_params, self))

        # Banks
        _params = (ExogenousFactors.bankSizeDistribution,
                   not ExogenousFactors.areBanksZeroIntelligenceAgents,
                   ExogenousFactors.DefaultEWADampingFactor)
        for _ in range(self.numberBanks):
            bank = Bank(*_params, self)
            self.schedule.add_bank(bank)
        self.normalize_banks()

        _params_depositors = (
            not ExogenousFactors.areDepositorsZeroIntelligenceAgents,
            ExogenousFactors.DefaultEWADampingFactor)

        # Depositors and Corporate Clients (Firms)
        if ExogenousFactors.standardCorporateClients:
            _params_corporate_clients = (ExogenousFactors.standardCorporateClientDefaultRate,
                                         ExogenousFactors.standardCorporateClientLossGivenDefault,
                                         ExogenousFactors.standardCorporateClientLoanInterestRate)
        else:
            _params_corporate_clients = (ExogenousFactors.wholesaleCorporateClientDefaultRate,
                                         ExogenousFactors.wholesaleCorporateClientLossGivenDefault,
                                         ExogenousFactors.wholesaleCorporateClientLoanInterestRate)

        for bank in self.schedule.banks:
            for i in range(ExogenousFactors.numberDepositorsPerBank):
                depositor = Depositor(*_params_depositors, bank, self)
                bank.depositors.append(depositor)
                self.schedule.add_depositor(depositor)
            for i in range(ExogenousFactors.numberCorporateClientsPerBank):
                corporate_client = CorporateClient(*_params_corporate_clients, bank, self)
                # bank.corporateClients.append(corporate_client)
                self.schedule.add_corporate_client(corporate_client)

    def step(self):
        self.schedule.reset_cycle()
        self.schedule.period_0()
        self.schedule.period_1()
        self.schedule.period_2()

    def run_model(self, n):
        for i in range(n):
            if i % 100 == 0:
                print(i)
            self.step()
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
            model_reporters={"Insolvencies": number_of_insolvencies,
                             "Contagions": number_of_contagions,
                             "Interbank_Loans": interbank_loans,
                             "Real_Sector_Loans": real_sector_loans,
                             "Real_Sector_Interest_Rate": real_sector_interest_rate,
                             'liquid_assets': liquid_assets,
                             'Early': early_withdrawal,
                             'credit_demand_fullfiled': credit_demand_fullfiled,
                             'credit_supply_exahausted': credit_supply_exahausted,
                             'beta_values': beta_values,
                             'beta_sd': beta_sd,
                             'interbank_interest_rate':interbank_interest_rate}#,

            #agent_reporters={"Beta":lambda a:a.currentlyChosenStrategy.get_beta_value() if a.is_bank else 0,
                           # "Interest_rate_real":lambda a:a.currentlyChosenStrategy.get_beta_value() if a.is_bank else 0}
        )

    def step(self):
        start = time.perf_counter()

        super().step()
        end = time.perf_counter()
        # print("Elapsed step = {}s".format((end - start)))
        # DataCollector
        self.datacollector.collect(self)


# Data Collector Functions
def number_of_insolvencies(model):
    return model.schedule.central_bank.insolvencyPerCycleCounter / model.numberBanks


def number_of_contagions(model):
    return model.schedule.central_bank.insolvencyDueToContagionPerCycleCounter / model.numberBanks


def real_sector_loans(model):
    return np.sum(model.schedule.real_sector_clearing_house.realSectorLendingMatrix)


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
    weights_matrix = lending_matrix / np.sum(lending_matrix)
    weighted_matrix = np.multiply(weights_matrix, interest_matrix)
    average_interest = np.sum(weighted_matrix)
    return average_interest

def interbank_interest_rate(model):
    lending_matrix = np.abs(model.schedule.clearing_house.interbankLendingMatrix)
    interest_matrix = model.schedule.clearing_house.interbankInterestMatrix
    print(interest_matrix)
    weights_matrix = lending_matrix / np.sum(lending_matrix)
    weighted_matrix = np.multiply(weights_matrix, interest_matrix)
    average_interest = np.sum(weighted_matrix)
    return average_interest


def liquid_assets(model):
    x = [i.balanceSheet.liquidAssets for i in model.schedule.banks]
    return sum(x)


def beta_values(model):
    x = [i.currentlyChosenStrategy.get_beta_value() for i in model.schedule.banks]
    return np.mean(x)


def beta_sd(model):
    x = [i.currentlyChosenStrategy.get_beta_value() for i in model.schedule.banks]
    return np.std(x)


def credit_demand_fullfiled(model):
    x = [i.creditDemandFulfilled for i in model.schedule.corporate_clients][:-1]
    return np.mean(x)


def credit_supply_exahausted(model):
    x = [i.creditSupplyExhausted for i in model.schedule.banks]
    return np.mean(x)


def early_withdrawal(model):
    x = [i.amountEarlyWithdraw for i in model.schedule.depositors]
    return sum(x)


if __name__ == "__main__":
    start = time.perf_counter()

    modelo = MyModel()
    modelo.run_model(1000)

    Modelo_original = modelo.datacollector.get_model_vars_dataframe()
    #data_agents = modelo.datacollector.get_agent_vars_dataframe()
    # Modelo_original
    Modelo_original.to_stata(r'output_Caso8_Desenv_Expansive.dta')
    #data_agents.to_stata(r'output_agents.dta')

    end = time.perf_counter()

    print("Elapsed total = {}s".format((end - start)))

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


def car_cb(model):
    return model.schedule.central_bank.minimumCapitalAdequacyRatio


"""
