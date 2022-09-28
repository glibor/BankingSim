# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:12:49 2021

@author: Douglas Silveira
"""

import pandas as pd
# import matplotlib.pyplot as plt
from mesa.datacollection import DataCollector
import statistics
from numba import jit
import numpy as np
import sys

# import random this is a test

# random.seed(1234)
np.random.seed(10)


class Util:
    id = 0

    @staticmethod
    def get_random_uniform(max_size):
        return np.random.uniform(0, max_size)

    @staticmethod
    def get_random_log_normal(mean, standard_deviation):
        return np.random.lognormal(mean, standard_deviation)

    @classmethod
    def get_unique_id(cls):
        cls.id += 1
        return cls.id


####################
##Exogenous factors#
####################

from enum import Enum


class SimulationType(Enum):
    General = 0
    HighSpread = 1
    LowSpread = 2
    ClearingHouse = 3
    ClearingHouseLowSpread = 4
    Basel = 5
    BaselBenchmark = 6
    DepositInsurance = 7
    DepositInsuranceBenchmark = 8
    RestrictiveMonetaryPolicy = 9
    ExpansiveMonetaryPolicy = 10


class BankSizeDistribution(Enum):
    Vanilla = 1
    LogNormal = 2


class InterbankPriority(Enum):
    Random = 1
    RiskSorted = 2


class ExogenousFactors:
    # Model
    numberBanks = 50
    depositInterestRate = 0.005
    interbankInterestRate = 0.005
    liquidAssetsInterestRate = 0
    illiquidAssetDiscountRate = 0.1
    interbankLendingMarketAvailable = True
    banksMaySellNonLiquidAssetsAtDiscountPrices = True
    banksHaveLimitedLiability = False

    # Banks
    bankSizeDistribution = BankSizeDistribution.LogNormal
    numberDepositorsPerBank = 100
    numberCorporateClientsPerBank = 50
    areBanksZeroIntelligenceAgents = False
    fireSellLowRisk = 0.01
    fireSellHighRisk = 0.03

    # Central Bank
    centralBankLendingInterestRate = 0.0575
    offersDiscountWindowLending = True
    minimumCapitalAdequacyRatio = 0.08
    isCentralBankZeroIntelligenceAgent = True
    isCapitalRequirementActive = True
    isTooBigToFailPolicyActive = False
    isDepositInsuranceAvailable = False
    isMonetaryPolicyAvailable = False

    # Clearing House
    isClearingGuaranteeAvailable = False
    interbankPriority = InterbankPriority.Random
    ##Para rankear por risco: InterbankPriority.RiskSorted
    ##Para rankear aleatório: InterbankPriority.Random

    # Depositors
    areDepositorsZeroIntelligenceAgents = True
    areBankRunsPossible = True
    amountWithdrawn = 1.0
    probabilityofWithdrawal = 0.1

    # Firms / Corporate Clients
    HighRiskCorporateClientDefaultRate = 0.1
    HighRiskCorporateClientLossGivenDefault = 0.7
    HighRiskCorporateClientLoanInterestRate = 0.065
    LowRiskCorporateClientDefaultRate = 0.01
    LowRiskCorporateClientLoanInterestRate = 0.035
    LowRiskCorporateClientLossGivenDefault = 0.6

    # Risk Weights
    CashRiskWeight = 0
    HighRiskCorporateLoanRiskWeight = 1.5
    InterbankLoanRiskWeight = 0.2
    LowRiskCorporateLoanRiskWeight = 0.5

    # Learning
    DefaultEWADampingFactor = 1


##############
# STRATEGIES###
##############

class BankEWAStrategy:
    # capital ratio (capital / assets)
    numberAlphaOptions = 30
    # liquidity ratio(liquid assets / deposits)
    numberBetaOptions = 30
    # risk appetite (High Risk Corporate Client/ Low Risk Corporate Client)
    numberGammaOptions = 30

    def __init__(self, alpha_index_option=0, beta_index_option=0, gamma_index_option=0):
        self.alphaIndex = alpha_index_option
        self.betaIndex = beta_index_option
        self.gammaIndex = gamma_index_option
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0

    def __repr__(self):
        return "BankEWAStragegy: alpha=" + str(self.get_alpha_value()) + ", " \
                                                                         "beta=" + str(self.get_beta_value()) + ", " \
                                                                                                                "gamma=" + str(
            self.get_gamma_value())

    def get_alpha_value(self):
        return (self.alphaIndex + 1) / 100

    def get_beta_value(self):
        return (self.betaIndex + 1) / 100

    def get_gamma_value(self):
        return (self.gammaIndex + 1) / 100

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.alphaIndex == other.alphaIndex and self.betaIndex == other.betaIndex and \
                   self.gammaIndex == other.gammaIndex
        return False

    def reset(self):
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0

    @classmethod
    def bank_ewa_strategy_list(cls):
        return [BankEWAStrategy(a, b, c) for a in range(cls.numberAlphaOptions) for b in \
                range(cls.numberBetaOptions) for c in range(cls.numberGammaOptions)]


import numpy as np


class CentralBankEWAStrategy:
    numberAlphaOptions = 10

    def __init__(self, alpha_index_option=0):
        self.alphaIndex = alpha_index_option
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0
        self.numberInsolvencies = self.totalLoans = 0

    def get_alpha_value(self):
        return (self.alphaIndex + 1) / 100

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.alphaIndex == other.alphaIndex
        return False

    def reset(self):
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0
        self.numberInsolvencies = self.totalLoans = 0

    @classmethod
    def central_bank_ewa_strategy_list(cls):
        return np.array([CentralBankEWAStrategy(a) for a in range(cls.numberAlphaOptions)],
                        dtype=CentralBankEWAStrategy)


class DepositorEWAStrategy:
    numberAlphaOptions = 10

    def __init__(self, alpha_index_option=0):
        self.alphaIndex = alpha_index_option
        self.strategyProfit = 0
        self.amountEarlyWithdraw = 0
        self.amountFinalWithdraw = 0
        self.insolvencyCounter = 0
        self.finalConsumption = 0
        self.A = self.P = self.F = 0

    def get_alpha_value(self):
        return (self.alphaIndex + 1) / 100

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.alphaIndex == other.alphaIndex
        return False

    def reset(self):
        self.strategyProfit = self.amountEarlyWithdraw = self.amountFinalWithdraw = 0
        self.insolvencyCounter = self.finalConsumption = 0
        self.A = self.P = self.F = 0

    @classmethod
    def depositor_ewa_strategy_list(cls):
        return np.array([DepositorEWAStrategy(a) for a in range(cls.numberAlphaOptions)], \
                        dtype=DepositorEWAStrategy)


##############
##AGENTS######
##############


##############
##BANKS#######
##############

from copy import copy

import numpy as np
from mesa import Agent


class Bank(Agent):

    def __init__(self, bank_size_distribution, is_intelligent, ewa_damping_factor, model):
        super().__init__(Util.get_unique_id(), model)

        self.initialSize = 1 if bank_size_distribution != BankSizeDistribution.LogNormal \
            else Util.get_random_log_normal(-0.5, 1)

        self.interbankHelper = InterbankHelper()
        self.guaranteeHelper = GuaranteeHelper()
        self.depositors = []  # Depositors
        self.LowRiskpoolcorporateClients = []  # Pool of LowRiskCorporateClients --> ActivationMethod povoa estas listas. Cada uma
        self.HighRiskpoolcorporateClients = []  # Pool of HighRiskCorporateClients         contêm 50 corporate clients.

        self.LowRiskcorporateClients = []  # LowRiskCorporateClients --> quero que os bancos povoem esta lista com um percentual da
        self.HighRiskcorporateClients = []  # HighRiskCorporateClients         anterior. O mesmo vale para a de baixo (γ define)

        self.quantityHighRiskcorporateClients = 0
        self.quantityLowRiskcorporateClients = 0

        self.liquidityNeeds = 0
        self.bankRunOccurred = False
        self.withdrawalsCounter = 0

        self.balanceSheet = BalanceSheet()
        self.auxBalanceSheet = None

        self.isIntelligent = is_intelligent
        if self.isIntelligent:
            self.strategiesOptionsInformation = BankEWAStrategy.bank_ewa_strategy_list()
            self.currentlyChosenStrategy = None
            self.EWADampingFactor = ewa_damping_factor

    def update_strategy_choice_probability(self):
        list_a = np.array([0.9999 * s.A + 0.005 * s.strategyProfitPercentageDamped for s
                           in self.strategiesOptionsInformation])
        # Caju input        
        # list_a = np.array([707 if x>707 else x for x in list_a])
        _exp = np.exp(list_a)
        list_p = _exp / np.sum(_exp)
        list_f = np.cumsum(list_p)
        for i, strategy in enumerate(self.strategiesOptionsInformation):
            strategy.A, strategy.P, strategy.F = list_a[i], list_p[i], list_f[i]

    def pick_new_strategy(self):
        probability_threshold = Util.get_random_uniform(1)
        self.currentlyChosenStrategy = [s for s in self.strategiesOptionsInformation if s.F > \
                                        probability_threshold][0]

    def reset(self):
        self.liquidityNeeds = 0
        self.bankRunOccurred = False
        self.withdrawalsCounter = 0

    def reset_collateral(self):
        self.guaranteeHelper = GuaranteeHelper()

    def setup_balance_sheet_intelligent(self, strategy=None):

        strategy = self.currentlyChosenStrategy
        risk_appetite = strategy.get_gamma_value()
        self.balanceSheet.liquidAssets = self.initialSize * strategy.get_beta_value()
        self.balanceSheet.nonFinancialSectorLoanHighRisk = (self.initialSize - \
                                                            self.balanceSheet.liquidAssets) * risk_appetite
        self.balanceSheet.nonFinancialSectorLoanLowRisk = self.initialSize - \
                                                          self.balanceSheet.liquidAssets - self.balanceSheet.nonFinancialSectorLoanHighRisk

        self.balanceSheet.interbankLoan = 0
        self.balanceSheet.discountWindowLoan = 0
        self.balanceSheet.deposits = self.initialSize * (strategy.get_alpha_value() - 1)
        self.liquidityNeeds = 0
        self.setup_balance_sheet()

    def choose_corporateClient(self, strategy=None):
        # strategy = self.currentlyChosenStrategy
        risk_appetite = self.currentlyChosenStrategy.get_gamma_value()
        self.quantityHighRiskcorporateClients = int(ExogenousFactors.numberCorporateClientsPerBank * \
                                                    risk_appetite)
        self.quantityLowRiskcorporateClients = ExogenousFactors.numberCorporateClientsPerBank - \
                                               self.quantityHighRiskcorporateClients

        self.HighRiskcorporateClients = \
            self.HighRiskpoolcorporateClients[0:self.quantityHighRiskcorporateClients + 1]

        self.LowRiskcorporateClients = \
            self.LowRiskpoolcorporateClients[0:self.quantityLowRiskcorporateClients + 1]

    def setup_balance_sheet(self):
        loan_per_coporate_clientLowRisk = self.balanceSheet.nonFinancialSectorLoanLowRisk \
                                          / len(self.LowRiskcorporateClients) if len(
            self.LowRiskcorporateClients) != 0 else 0

        loan_per_coporate_clientHighRisk = self.balanceSheet.nonFinancialSectorLoanHighRisk \
                                           / len(self.HighRiskcorporateClients) if len(
            self.HighRiskcorporateClients) != 0 else 0

        for corporateClient in self.LowRiskcorporateClients:
            corporateClient.loanAmount = loan_per_coporate_clientLowRisk

        for corporateClient in self.HighRiskcorporateClients:
            corporateClient.loanAmount = loan_per_coporate_clientHighRisk

        deposit_per_depositor = -self.balanceSheet.deposits / len(self.depositors)
        for depositor in self.depositors:
            depositor.make_deposit(deposit_per_depositor)

    def get_capital_adequacy_ratio(self):
        if self.is_solvent():
            rwa = self.get_real_sector_risk_weighted_assets()
            total_risk_weighted_assets = self.balanceSheet.liquidAssets * ExogenousFactors.CashRiskWeight \
                                         + rwa

            if self.is_interbank_creditor():
                total_risk_weighted_assets += self.balanceSheet.interbankLoan * \
                                              ExogenousFactors.InterbankLoanRiskWeight

            if total_risk_weighted_assets != 0:
                return -self.balanceSheet.capital / total_risk_weighted_assets
        return 0

    def adjust_capital_ratio(self, minimum_capital_ratio_required):
        current_capital_ratio = self.get_capital_adequacy_ratio()
        if current_capital_ratio <= minimum_capital_ratio_required:
            adjustment_factor = current_capital_ratio / minimum_capital_ratio_required

            for corporateClient in self.LowRiskcorporateClients:
                original_loan_amount = corporateClient.loanAmount
                new_loan_amount = original_loan_amount * adjustment_factor
                corporateClient.loanAmount = new_loan_amount
                self.balanceSheet.liquidAssets += (original_loan_amount - new_loan_amount)

            for corporateClient in self.HighRiskcorporateClients:
                original_loan_amount2 = corporateClient.loanAmount
                new_loan_amount2 = original_loan_amount2 * adjustment_factor
                corporateClient.loanAmount = new_loan_amount2
                self.balanceSheet.liquidAssets += (original_loan_amount2 - new_loan_amount2)

            self.update_non_financial_sector_loans()

    def update_non_financial_sector_loans(self):
        self.balanceSheet.nonFinancialSectorLoanLowRisk = sum(
            client.loanAmount for client in self.LowRiskcorporateClients)

        self.balanceSheet.nonFinancialSectorLoanHighRisk = sum(
            client.loanAmount for client in self.HighRiskcorporateClients)

    def get_real_sector_risk_weighted_assets(self):
        x = self.balanceSheet.nonFinancialSectorLoanLowRisk * ExogenousFactors.LowRiskCorporateLoanRiskWeight
        y = self.balanceSheet.nonFinancialSectorLoanHighRisk * ExogenousFactors.HighRiskCorporateLoanRiskWeight

        return x + y

    def withdraw_deposit(self, amount_to_withdraw):
        if amount_to_withdraw > 0:
            self.withdrawalsCounter += 1
        self.liquidityNeeds -= amount_to_withdraw
        return amount_to_withdraw

    def use_liquid_assets_to_pay_depositors_back(self):
        if self.needs_liquidity():
            original_liquid_assets = self.balanceSheet.liquidAssets
            self.liquidityNeeds += original_liquid_assets
            self.balanceSheet.liquidAssets = max(self.liquidityNeeds, 0)
            resulting_liquid_assets = self.balanceSheet.liquidAssets
            total_paid = original_liquid_assets - resulting_liquid_assets
            self.balanceSheet.deposits += total_paid

    def accrue_interest_balance_sheet(self):
        self.balanceSheet.discountWindowLoan *= (1 + ExogenousFactors.centralBankLendingInterestRate)
        self.balanceSheet.liquidAssets *= (1 + self.model.liquidAssetsInterestRate)
        self.calculate_deposits_interest()

    def calculate_deposits_interest(self):
        deposits_interest_rate = 1 + self.model.depositInterestRate
        self.balanceSheet.deposits *= deposits_interest_rate
        for depositor in self.depositors:
            depositor.deposit.amount *= deposits_interest_rate

    def collect_loans(self):
        self.balanceSheet.nonFinancialSectorLoanLowRisk = sum(
            client.pay_loan_back() for client in self.LowRiskcorporateClients)

        self.balanceSheet.nonFinancialSectorLoanHighRisk = sum(
            client.pay_loan_back() for client in self.HighRiskcorporateClients)

    def offers_liquidity(self):
        return self.liquidityNeeds > 0

    def needs_liquidity(self):
        return self.liquidityNeeds <= 0

    def is_liquid(self):
        return self.liquidityNeeds >= 0

    def receive_discount_window_loan(self, amount):
        self.balanceSheet.discountWindowLoan = amount
        self.balanceSheet.deposits -= amount
        self.liquidityNeeds -= amount

    def use_non_liquid_assets_to_pay_depositors_back(self):
        if self.needs_liquidity():
            total_loans = self.balanceSheet.nonFinancialSectorLoanLowRisk + \
                          self.balanceSheet.nonFinancialSectorLoanHighRisk
            liquidity_needed = -self.liquidityNeeds
            total_loans_to_sell = liquidity_needed * (1 + ExogenousFactors.illiquidAssetDiscountRate)

            if total_loans > total_loans_to_sell:

                proportionLoanLowRisk = self.balanceSheet.nonFinancialSectorLoanLowRisk / (
                        self.balanceSheet.nonFinancialSectorLoanLowRisk + self.balanceSheet.nonFinancialSectorLoanHighRisk)
                amountSoldLowRisk = total_loans_to_sell * proportionLoanLowRisk
                amountSoldHighRisk = total_loans_to_sell * (1 - proportionLoanLowRisk)
                self.liquidityNeeds = 0
                self.balanceSheet.deposits += liquidity_needed
                proportion_of_illiquid_assets_sold_low_risk = amountSoldLowRisk / \
                                                              self.balanceSheet.nonFinancialSectorLoanLowRisk
                proportion_of_illiquid_assets_sold_high_risk = amountSoldHighRisk / \
                                                               self.balanceSheet.nonFinancialSectorLoanHighRisk
                for firm in self.LowRiskcorporateClients:
                    firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold_low_risk
                for firm in self.HighRiskcorporateClients:
                    firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold_high_risk

                if (self.balanceSheet.nonFinancialSectorLoanLowRisk >= amountSoldLowRisk):
                    self.balanceSheet.nonFinancialSectorLoanLowRisk -= amountSoldLowRisk
                else:
                    self.balanceSheet.nonFinancialSectorLoanLowRisk = 0

                if (self.balanceSheet.nonFinancialSectorLoanHighRisk >= amountSoldHighRisk):
                    self.balanceSheet.nonFinancialSectorLoanHighRisk -= amountSoldHighRisk
                else:
                    self.balanceSheet.nonFinancialSectorLoanHighRisk = 0

            # if total_loans > total_loans_to_sell:
            #     # Naturalmente fica mais arriscado, pois vende os ativos menos arriscados
            #     # Deveria ter feito igual para todos...
            #     if self.balanceSheet.nonFinancialSectorLoanLowRisk >= total_loans_to_sell:
            #         amount_sold = total_loans_to_sell
            #         self.liquidityNeeds = 0
            #         self.balanceSheet.deposits += liquidity_needed

            #         proportion_of_illiquid_assets_sold = amount_sold / \
            #         self.balanceSheet.nonFinancialSectorLoanLowRisk

            #         for firm in self.LowRiskcorporateClients:
            #             firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold

            #         self.balanceSheet.nonFinancialSectorLoanLowRisk -= amount_sold

            #     else:
            #         x = total_loans_to_sell - self.balanceSheet.nonFinancialSectorLoanLowRisk
            #         amount_sold = self.balanceSheet.nonFinancialSectorLoanLowRisk + \
            #         (self.balanceSheet.nonFinancialSectorLoanHighRisk - x)

            #         self.liquidityNeeds = 0
            #         self.balanceSheet.deposits += liquidity_needed
            #         proportion_of_illiquid_assets_sold_HighRisk = x / \
            #         self.balanceSheet.nonFinancialSectorLoanHighRisk

            #         for firm in self.LowRiskcorporateClients:
            #             firm.loanAmount *= 0
            #         for firm in self.HighRiskcorporateClients:
            #             firm.loanAmount *= (1 - proportion_of_illiquid_assets_sold_HighRisk)

            #         self.balanceSheet.nonFinancialSectorLoanLowRisk = 0
            #         self.balanceSheet.nonFinancialSectorLoanHighRisk -= x

            else:
                amount_sold = total_loans
                self.liquidityNeeds += amount_sold / (1 + ExogenousFactors.illiquidAssetDiscountRate)
                self.balanceSheet.deposits += liquidity_needed - self.liquidityNeeds

                # Nesse caso vai ser sempre 1
                proportion_of_illiquid_assets_sold = amount_sold / total_loans

                for firm in self.LowRiskcorporateClients:
                    firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold
                for firm in self.HighRiskcorporateClients:
                    firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold

                self.balanceSheet.nonFinancialSectorLoanLowRisk = 0
                self.balanceSheet.nonFinancialSectorLoanHighRisk = 0

    def get_profit(self):
        resulting_capital = self.balanceSheet.assets + self.balanceSheet.liabilities
        original_capital = self.auxBalanceSheet.assets + self.auxBalanceSheet.liabilities
        if ExogenousFactors.banksHaveLimitedLiability:
            resulting_capital = max(resulting_capital, 0)
        return resulting_capital - original_capital

    def calculate_profit(self, minimum_capital_ratio_required):
        if self.isIntelligent:
            strategy = self.currentlyChosenStrategy

            self.bankRunOccurred = (self.withdrawalsCounter > ExogenousFactors.numberDepositorsPerBank / 2)

            if self.bankRunOccurred:
                original_loans = self.auxBalanceSheet.nonFinancialSectorLoanLowRisk + \
                                 self.auxBalanceSheet.nonFinancialSectorLoanHighRisk

                resulting_loans = self.balanceSheet.nonFinancialSectorLoanLowRisk + \
                                  self.balanceSheet.nonFinancialSectorLoanHighRisk

                delta = original_loans - resulting_loans
                if delta > 0:
                    self.balanceSheet.nonFinancialSectorLoanLowRisk -= (delta * ExogenousFactors.fireSellLowRisk)
                    self.balanceSheet.nonFinancialSectorLoanHighRisk -= (delta * ExogenousFactors.fireSellHighRisk)

            profit = self.get_profit()

            strategy.strategyProfit = profit

            if ExogenousFactors.isCapitalRequirementActive:
                current_capital_ratio = self.get_capital_adequacy_ratio()

                if current_capital_ratio < minimum_capital_ratio_required:
                    delta_capital_ratio = minimum_capital_ratio_required - current_capital_ratio
                    strategy.strategyProfit -= delta_capital_ratio

            # Return on Equity, based on initial shareholders equity.
            strategy.strategyProfitPercentage = -strategy.strategyProfit / self.auxBalanceSheet.capital
            strategy.strategyProfitPercentageDamped = \
                strategy.strategyProfitPercentage * self.EWADampingFactor

    def liquidate(self):
        #  first, sell assets...
        self.balanceSheet.liquidAssets += (self.balanceSheet.nonFinancialSectorLoanLowRisk + \
                                           self.balanceSheet.nonFinancialSectorLoanHighRisk)
        self.balanceSheet.nonFinancialSectorLoanLowRisk = 0
        self.balanceSheet.nonFinancialSectorLoanHighRisk = 0

        if self.is_interbank_creditor():
            self.balanceSheet.liquidAssets += self.balanceSheet.interbankLoan
            self.balanceSheet.interbankLoan = 0

        # then, resolve liabilities, in order of subordination...

        #  ...1st, discountWindowLoan...
        if self.balanceSheet.liquidAssets > abs(self.balanceSheet.discountWindowLoan):
            self.balanceSheet.liquidAssets += self.balanceSheet.discountWindowLoan
            self.balanceSheet.discountWindowLoan = 0
        else:
            self.balanceSheet.liquidAssets = 0
            self.balanceSheet.discountWindowLoan += self.balanceSheet.liquidAssets

        # ...2nd, interbank loans...
        if self.is_interbank_debtor():
            if self.balanceSheet.liquidAssets > abs(self.balanceSheet.interbankLoan):
                self.balanceSheet.interbankLoan = 0
                self.balanceSheet.liquidAssets += self.balanceSheet.interbankLoan
            else:
                self.balanceSheet.liquidAssets = 0
                self.balanceSheet.interbankLoan += self.balanceSheet.liquidAssets

        # ... finally, if there is any money left, it is proportionally divided among depositors.
        percentage_deposits_payable = self.balanceSheet.liquidAssets / \
                                      np.absolute(self.balanceSheet.deposits)
        self.balanceSheet.deposits *= percentage_deposits_payable

        for depositor in self.depositors:
            depositor.deposit.amount *= percentage_deposits_payable

        self.balanceSheet.liquidAssets = 0

    def is_insolvent(self):
        return self.balanceSheet.capital > 0

    def is_solvent(self):
        return self.balanceSheet.capital <= 0

    def is_interbank_creditor(self):
        return self.balanceSheet.interbankLoan >= 0

    def is_interbank_debtor(self):
        return self.balanceSheet.interbankLoan < 0

    def period_0(self):
        if self.isIntelligent:
            self.update_strategy_choice_probability()
            self.pick_new_strategy()
            self.choose_corporateClient()
            self.setup_balance_sheet_intelligent(self.currentlyChosenStrategy)
        else:
            self.choose_corporateClient()
            self.setup_balance_sheet()
        self.auxBalanceSheet = copy(self.balanceSheet)

    def period_1(self):
        # First, banks try to use liquid assets to pay early withdrawals...
        self.use_liquid_assets_to_pay_depositors_back()
        # ... if needed, they will try interbank market by clearing house.
        # ... if banks still needs liquidity, central bank might rescue...

    def period_2(self):
        self.accrue_interest_balance_sheet()
        self.collect_loans()


class InterbankHelper:
    def __init__(self):
        self.counterpartyID = 0
        self.priorityOrder = 0
        self.auxPriorityOrder = 0
        self.loanAmount = 0
        self.acumulatedLiquidity = 0
        self.riskSorting = None
        self.amountLiquidityLeftToBorrowOrLend = 0


class GuaranteeHelper:
    def __init__(self):
        self.potentialCollateral = 0
        self.feasibleCollateral = 0
        self.outstandingAmountImpact = 0
        self.residual = 0
        self.redistributedCollateral = 0
        self.collateralAdjustment = 0


class BalanceSheet:
    def __init__(self):
        self.deposits = 0
        self.discountWindowLoan = 0
        self.interbankLoan = 0
        self.nonFinancialSectorLoanLowRisk = 0
        self.nonFinancialSectorLoanHighRisk = 0
        self.liquidAssets = 0

    @property
    def capital(self):
        return -(self.liquidAssets +
                 self.nonFinancialSectorLoanLowRisk +
                 self.nonFinancialSectorLoanHighRisk +
                 self.interbankLoan +
                 self.discountWindowLoan +
                 self.deposits)

    @property
    def assets(self):
        return self.liquidAssets + self.nonFinancialSectorLoanLowRisk + \
               self.nonFinancialSectorLoanHighRisk + np.max(self.interbankLoan, 0)

    @property
    def liabilities(self):
        return self.deposits + self.discountWindowLoan + np.min(self.interbankLoan, 0)


################
##CENTRAL BANK##
################

import numpy as np
from mesa import Agent


class CentralBank(Agent):

    def __init__(self, central_bank_lending_interest_rate, offers_discount_window_lending,
                 minimum_capital_adequacy_ratio, is_intelligent, ewa_damping_factor, model):
        super().__init__(Util.get_unique_id(), model)

        self.centralBankLendingInterestRate = central_bank_lending_interest_rate
        self.offersDiscountWindowLending = offers_discount_window_lending
        self.minimumCapitalAdequacyRatio = minimum_capital_adequacy_ratio

        self.insolvencyPerCycleCounter = 0
        self.insolvencyDueToContagionPerCycleCounter = 0

        self.isIntelligent = is_intelligent
        if self.isIntelligent:
            self.strategiesOptionsInformation = CentralBankEWAStrategy.central_bank_ewa_strategy_list()
            self.currentlyChosenStrategy = None
            self.EWADampingFactor = ewa_damping_factor

    def update_strategy_choice_probability(self):
        list_a = np.array([0.9999 * s.A + 0.005 * s.strategyProfit for s in self.strategiesOptionsInformation])
        _exp = np.exp(list_a)
        list_p = _exp / np.sum(_exp)
        list_f = np.cumsum(list_p)
        for i, strategy in enumerate(self.strategiesOptionsInformation):
            strategy.A, strategy.P, strategy.F = list_a[i], list_p[i], list_f[i]

    def pick_new_strategy(self):
        probability_threshold = Util.get_random_uniform(1)
        self.currentlyChosenStrategy = [s for s in self.strategiesOptionsInformation if s.F > \
                                        probability_threshold][0]

    def observe_banks_capital_adequacy(self, banks):
        for bank in banks:
            if bank.get_capital_adequacy_ratio() < self.minimumCapitalAdequacyRatio:
                bank.adjust_capital_ratio(self.minimumCapitalAdequacyRatio)

    def organize_discount_window_lending(self, banks):
        for bank in banks:
            if not bank.is_liquid():
                loan_amount = self.get_discount_window_lend(bank, bank.liquidityNeeds)
                bank.receive_discount_window_loan(loan_amount)

    def get_discount_window_lend(self, bank, amount_needed):
        # when should not bank be eligible for such loans?
        if self.offersDiscountWindowLending:
            if ExogenousFactors.isTooBigToFailPolicyActive:
                if CentralBank.is_bank_too_big_to_fail(bank):
                    return min(amount_needed, 0)
                else:
                    return 0  # better luck next time!
            else:
                # If Central Bank offers lending and TBTF is not active, assume all banks get help
                return min(amount_needed, 0)
        else:
            return 0

    @staticmethod
    def is_bank_too_big_to_fail(bank):
        if ExogenousFactors.isTooBigToFailPolicyActive:
            random_uniform = Util.get_random_uniform(1)
            return random_uniform < 2 * bank.marketShare
        return False

    @staticmethod
    def make_banks_sell_non_liquid_assets(banks):
        for bank in banks:
            if not bank.is_liquid():
                bank.use_non_liquid_assets_to_pay_depositors_back()

    @staticmethod
    def bailout(bank):
        if not bank.is_liquid():
            liquidity_needs = -bank.liquidityNeeds
            bank.balanceSheet.liquidAssets += liquidity_needs
            bank.liquidityNeeds = 0
        if bank.is_insolvent():
            capital_shortfall = bank.balanceSheet.capital
            bank.balanceSheet.liquidAssets += capital_shortfall

    @staticmethod
    def punish_illiquidity(bank):
        # is there anything else to do?
        bank.use_non_liquid_assets_to_pay_depositors_back()

    def punish_insolvency(self, bank):
        insolvency_penalty_LowRisk = 0.5
        insolvency_penalty_HighRisk = 0.8

        bank.balanceSheet.nonFinancialSectorLoanLowRisk *= 1 - insolvency_penalty_LowRisk
        bank.balanceSheet.nonFinancialSectorLoanHighRisk *= 1 - insolvency_penalty_HighRisk
        self.insolvencyPerCycleCounter += 1

    def punish_contagion_insolvency(self, bank):
        self.insolvencyDueToContagionPerCycleCounter += 1
        self.punish_insolvency(bank)

    def calculate_final_utility(self, banks):
        if self.isIntelligent:
            strategy = self.currentlyChosenStrategy
            strategy.numberInsolvencies = self.insolvencyPerCycleCounter
            strategy.totalLoans = CentralBank.get_total_real_sector_loans(banks)
            potential_total_size = len(banks)
            ratio = strategy.totalLoans / potential_total_size
            strategy.strategyProfit = ratio - (potential_total_size * strategy.numberInsolvencies)

    @staticmethod
    def get_total_real_sector_loans(banks):
        return sum([bank.balanceSheet.nonFinancialSectorLoanLowRisk for bank in banks]) + \
               sum([bank.balanceSheet.nonFinancialSectorLoanHighRisk for bank in banks])

    @staticmethod
    def liquidate_insolvent_banks(banks):
        for bank in banks:
            if bank.is_insolvent():
                bank.liquidate()

    @property
    def banks(self):
        return self.model.schedule.banks

    def reset(self):
        self.insolvencyPerCycleCounter = 0
        self.insolvencyDueToContagionPerCycleCounter = 0

    def period_0(self):
        if self.isIntelligent:
            self.update_strategy_choice_probability()
            self.pick_new_strategy()
            self.minimumCapitalAdequacyRatio = self.currentlyChosenStrategy.get_alpha_value()
        if ExogenousFactors.isCapitalRequirementActive:
            self.observe_banks_capital_adequacy(self.banks)

    def period_1(self):
        # ... if banks still needs liquidity, central bank might rescue...
        if self.offersDiscountWindowLending:
            self.organize_discount_window_lending(self.banks)
        # ... if everything so far isn't enough, banks will sell illiquid assets at discount prices.
        if ExogenousFactors.banksMaySellNonLiquidAssetsAtDiscountPrices:
            CentralBank.make_banks_sell_non_liquid_assets(self.banks)

    def period_2(self):
        for bank in self.banks:
            if CentralBank.is_bank_too_big_to_fail(bank):
                CentralBank.bailout(bank)
            if not bank.is_liquid():
                CentralBank.punish_illiquidity(bank)
            if not bank.is_solvent():
                self.punish_insolvency(bank)

        if self.model.interbankLendingMarketAvailable:
            self.model.schedule.clearing_house.interbank_contagion(self.banks, self)

        for bank in self.banks:
            bank.calculate_profit(self.minimumCapitalAdequacyRatio)

        self.calculate_final_utility(self.banks)
        CentralBank.liquidate_insolvent_banks(self.banks)

        for depositor in self.model.schedule.depositors:
            depositor.calculate_final_utility()


######################
####CLEARING HOUSE###
#####################

import numpy as np
from mesa import Agent


class ClearingHouse(Agent):

    def __init__(self, number_banks, clearing_guarantee_available, model):
        super().__init__(Util.get_unique_id(), model)
        self.numberBanks = number_banks
        self.clearingGuaranteeAvailable = clearing_guarantee_available

        self.biggestInterbankDebt = 0
        self.totalInterbankDebt = 0
        self.totalCollateralDeficit = 0
        self.totalCollateralSurplus = 0

        self.interbankLendingMatrix = np.zeros((self.numberBanks, self.numberBanks))
        self.vetor_recuperacao = np.ones(self.numberBanks)
        # worst case scenario...
        self.banksNeedingLiquidity = list()
        self.banksOfferingLiquidity = list()

    def reset(self):
        self.interbankLendingMatrix[:, :] = 0
        self.reset_vetor_recuperacao()
        self.biggestInterbankDebt = 0
        self.totalInterbankDebt = 0
        self.totalCollateralDeficit = 0
        self.totalCollateralSurplus = 0
        self.banksNeedingLiquidity.clear()
        self.banksOfferingLiquidity.clear()

    def reset_vetor_recuperacao(self):
        self.vetor_recuperacao[:] = 1

    def organize_interbank_market_common(self, banks, simulation=False, m=0, simulated_strategy=None):
        for bank in self.model.schedule.banks:
            bank.interbankHelper.amountLiquidityLeftToBorrowOrLend = bank.liquidityNeeds
            if bank.needs_liquidity():
                self.banksNeedingLiquidity.append(bank)
            else:
                self.banksOfferingLiquidity.append(bank)

        if ExogenousFactors.interbankPriority == InterbankPriority.Random:
            np.random.shuffle(self.banksOfferingLiquidity)
            np.random.shuffle(self.banksNeedingLiquidity)
        elif ExogenousFactors.interbankPriority == InterbankPriority.RiskSorted:
            self.sort_queues_by_risk(simulation, m, simulated_strategy)

        for i, bank in enumerate(self.banksOfferingLiquidity):
            bank.interbankHelper.priorityOrder = i

        for i, bank in enumerate(self.banksNeedingLiquidity):
            bank.interbankHelper.priorityOrder = i

        iterator_lenders = iter(self.banksOfferingLiquidity)
        iterator_borrowers = iter(self.banksNeedingLiquidity)

        if len(self.banksOfferingLiquidity) > 0 and len(self.banksNeedingLiquidity) > 0:
            lender = next(iterator_lenders)
            borrower = next(iterator_borrowers)
            while True:
                try:
                    amount_offered = lender.interbankHelper.amountLiquidityLeftToBorrowOrLend
                    amount_requested = abs(borrower.interbankHelper.amountLiquidityLeftToBorrowOrLend)
                    amount_lent = min(amount_offered, amount_requested)
                    lender.interbankHelper.amountLiquidityLeftToBorrowOrLend -= amount_lent
                    borrower.interbankHelper.amountLiquidityLeftToBorrowOrLend += amount_lent

                    lender_id = (lender.unique_id - self.numberBanks) % self.numberBanks
                    borrower_id = (borrower.unique_id - self.numberBanks) % self.numberBanks

                    self.interbankLendingMatrix[lender_id, borrower_id] = amount_lent
                    self.interbankLendingMatrix[borrower_id, lender_id] = -amount_lent

                    if lender.interbankHelper.amountLiquidityLeftToBorrowOrLend == 0:
                        lender = next(iterator_lenders)
                    if borrower.interbankHelper.amountLiquidityLeftToBorrowOrLend == 0:
                        borrower = next(iterator_borrowers)
                except StopIteration:
                    break

        for bank in banks:
            bank.balanceSheet.interbankLoan = self.get_interbank_market_position(bank)

            if bank.offers_liquidity():
                # if there is any amount left offered, assign it to liquid assets
                bank.balanceSheet.liquidAssets = bank.interbankHelper.amountLiquidityLeftToBorrowOrLend
                bank.interbankHelper.amountLiquidityLeftToBorrowOrLend = 0

            if not bank.is_interbank_creditor():
                # if bank used interbank loan to pay depositors back, adjust deposit account
                bank.balanceSheet.deposits -= bank.balanceSheet.interbankLoan

            bank.liquidityNeeds = bank.interbankHelper.amountLiquidityLeftToBorrowOrLend

    def get_interbank_market_position(self, bank):
        bank_id_adjusted = (bank.unique_id - self.numberBanks) % self.numberBanks
        return np.sum(self.interbankLendingMatrix[bank_id_adjusted, :])

    def sort_queues_by_risk(self, simulation, bank_id_simulating, strategy_simulated):

        def bank_to_alpha_beta_gamma(_bank):
            strategy = _bank.interbankHelper.riskSorting
            return strategy.get_alpha_value(), strategy.get_beta_value(), strategy.get_gamma_value()

        for bank in self.banksOfferingLiquidity:
            if simulation and bank.unique_id == bank_id_simulating:
                bank.interbankHelper.riskSorting = strategy_simulated
            else:
                bank.interbankHelper.riskSorting = bank.currentlyChosenStrategy

        for bank in self.banksNeedingLiquidity:
            if simulation and bank.unique_id == bank_id_simulating:
                bank.interbankHelper.riskSorting = strategy_simulated
            else:
                bank.interbankHelper.riskSorting = bank.currentlyChosenStrategy

        self.banksOfferingLiquidity.sort(reverse=True, key=bank_to_alpha_beta_gamma)
        self.banksNeedingLiquidity.sort(reverse=True, key=bank_to_alpha_beta_gamma)

    def interbank_clearing_guarantee(self, banks):
        self.calculate_total_and_biggest_interbank_debt(banks)
        self.organize_guarantees(banks)

    def calculate_total_and_biggest_interbank_debt(self, banks):
        for bank in banks:
            if bank.balanceSheet.interbankLoan < self.biggestInterbankDebt:
                self.biggestInterbankDebt = bank.balanceSheet.interbankLoan

            if bank.balanceSheet.interbankLoan < 0:
                self.totalInterbankDebt = self.totalInterbankDebt - bank.balanceSheet.interbankLoan

    def organize_guarantees(self, banks):

        for bank in banks:
            bank.reset_collateral()
            if not bank.is_interbank_creditor():
                g_helper = bank.guaranteeHelper
                ratio = bank.balanceSheet.interbankLoan / self.totalInterbankDebt
                g_helper.potentialCollateral = self.biggestInterbankDebt * ratio
                # both assets can be used as collateral
                g_helper.feasibleCollateral = min(
                    g_helper.potentialCollateral,
                    bank.balanceSheet.liquidAssets + bank.balanceSheet.nonFinancialSectorLoanLowRisk + \
                    +bank.balanceSheet.nonFinancialSectorLoanHighRisk)
                # minimize to avoid insolvent bank to use collateral
                g_helper.feasibleCollateral = min(
                    g_helper.feasibleCollateral,
                    max(0, -bank.balanceSheet.interbankLoan - min(0, bank.balanceSheet.capital)))

                # interbank debit balance impact
                g_helper.outstandingAmountImpact = max(
                    0,
                    min(
                        bank.balanceSheet.capital + g_helper.feasibleCollateral,
                        -bank.balanceSheet.interbankLoan))
                # residual collateral
                g_helper.residual = g_helper.feasibleCollateral - g_helper.outstandingAmountImpact

        for bank in banks:

            g_helper = bank.guaranteeHelper

            # total of collateral deficit or surplus
            if g_helper.residual < 0:
                self.totalCollateralDeficit += g_helper.residual
            else:
                self.totalCollateralSurplus += g_helper.residual

        for bank in banks:

            g_helper = bank.guaranteeHelper

            # residual collateral redistributed
            if g_helper.residual < 0:
                g_helper.redistributedCollateral = g_helper.residual
            elif self.totalCollateralSurplus == 0:
                g_helper.redistributedCollateral = 0
            else:
                f = min(1.0, -self.totalCollateralDeficit / self.totalCollateralSurplus)
                g_helper.redistributedCollateral = (1 - f) * g_helper.residual

            # final total collateral
            g_helper.collateralAdjustment = g_helper.outstandingAmountImpact + g_helper.redistributedCollateral
            collateral = g_helper.feasibleCollateral - g_helper.collateralAdjustment

            bank.balanceSheet.nonFinancialSectorLoanLowRisk += \
                - max(0, collateral - bank.balanceSheet.liquidAssets) * 0.5

            bank.balanceSheet.nonFinancialSectorLoanHighRisk += \
                - max(0, collateral - bank.balanceSheet.liquidAssets) * 0.5

            bank.balanceSheet.liquidAssets += -min(bank.balanceSheet.liquidAssets, collateral)

    def interbank_contagion(self, banks, central_bank):
        self.reset_vetor_recuperacao()
        for bank in banks:

            bank_id = (bank.unique_id - self.numberBanks) % self.numberBanks

            if not bank.is_solvent() and bank.is_interbank_debtor():
                if self.clearingGuaranteeAvailable:
                    _max = max(0, -self.totalCollateralDeficit - self.totalCollateralSurplus)
                    self.vetor_recuperacao[bank_id] = (self.totalInterbankDebt + _max) / self.totalInterbankDebt
                else:
                    self.vetor_recuperacao[bank_id] = (bank.balanceSheet.interbankLoan + min(
                        -bank.balanceSheet.interbankLoan,
                        bank.balanceSheet.capital)) / bank.balanceSheet.interbankLoan

        for i in range(self.numberBanks):
            for j in range(i, self.numberBanks):
                self.interbankLendingMatrix[i, j] *= self.vetor_recuperacao[j]
                self.interbankLendingMatrix[j, i] = -self.interbankLendingMatrix[i, j]

        for bank in banks:
            bank_id = (bank.unique_id - self.numberBanks) % self.numberBanks
            bank.balanceSheet.interbankLoan = np.sum(self.interbankLendingMatrix[bank_id, :])
            if bank.is_insolvent():
                central_bank.punish_contagion_insolvency(bank)

    def accrue_interest(self, banks, interbank_rate):
        np.multiply(self.interbankLendingMatrix, (1 + interbank_rate), out=self.interbankLendingMatrix)
        for bank in banks:
            bank.balanceSheet.interbankLoan = self.get_interbank_market_position(bank)

    def period_0(self):
        pass

    def period_1(self):
        if self.model.interbankLendingMarketAvailable:
            self.organize_interbank_market_common(self.model.schedule.banks)
            if self.clearingGuaranteeAvailable:
                self.interbank_clearing_guarantee(self.model.schedule.banks)

    def period_2(self):
        self.accrue_interest(self.model.schedule.banks, self.model.interbankInterestRate)


######################
###CORPORATE CLIENT###
######################

from mesa import Agent


class CorporateClient(Agent):

    def __init__(self, default_rate, loss_given_default, loan_interest_rate, bank, model):
        super().__init__(Util.get_unique_id(), model)

        # Bank Reference
        self.bank = bank

        self.loanAmount = 0
        self.percentageRepaid = 0

        self.probabilityOfDefault = default_rate
        self.lossGivenDefault = loss_given_default
        self.loanInterestRate = loan_interest_rate

    def pay_loan_back(self, simulation=False):
        if simulation:
            # if under simulation, assume last percetageRepaid used
            amount_paid = self.percentageRepaid * self.loanAmount
        else:
            amount_paid = self.loanAmount * (1 - self.lossGivenDefault) \
                if Util.get_random_uniform(1) <= self.probabilityOfDefault \
                else self.loanAmount * (1 + self.loanInterestRate)
            self.percentageRepaid = 0 if self.loanAmount == 0 else amount_paid / self.loanAmount

        self.loanAmount = amount_paid
        return amount_paid

    def reset(self):
        self.loanAmount = 0
        self.percentageRepaid = 0

    def period_0(self):
        pass

    def period_1(self):
        pass

    def period_2(self):
        pass


######################
###DEPOSITOR########
#####################

import math

import numpy as np
from mesa import Agent


class Depositor(Agent):

    def __init__(self, is_intelligent, ewa_damping_factor, bank, model):
        super().__init__(Util.get_unique_id(), model)

        # Bank Reference
        self.bank = bank

        self.amountEarlyWithdraw = 0
        self.amountFinalWithdraw = 0
        self.safetyTreshold = 0

        self.profit = 0

        self.initialDeposit = Deposit()
        self.deposit = self.initialDeposit

        self.isIntelligent = is_intelligent
        if self.isIntelligent:
            self.strategiesOptionsInformation = DepositorEWAStrategy.depositor_ewa_strategy_list()
            self.currentlyChosenStrategy = None
            self.EWADampingFactor = ewa_damping_factor

    def update_strategy_choice_probability(self):
        list_a = np.array([s.A + 0.005 * s.strategyProfit for s in self.strategiesOptionsInformation])
        # Caju input        
        # list_a = np.array([707 if x>707 else x for x in list_a])
        _exp = np.exp(list_a)
        list_p = _exp / np.sum(_exp)
        list_f = np.cumsum(list_p)
        for i, strategy in enumerate(self.strategiesOptionsInformation):
            strategy.A, strategy.P, strategy.F = list_a[i], list_p[i], list_f[i]

    def pick_new_strategy(self):
        probability_threshold = Util.get_random_uniform(1)
        try:
            self.currentlyChosenStrategy = [s for s in self.strategiesOptionsInformation if s.F > \
                                            probability_threshold][0]
        except:

            #  new_A = np.array([700 if s.A>700 else s.A for s in self.strategiesOptionsInformation])
            # new_exp = np.exp(new_A)
            # new_p = new_exp / np.sum(new_exp)
            # new_f = np.cumsum(new_p)
            # print('threshold',probability_threshold)
            print('s.a', [s.A for s in self.strategiesOptionsInformation])
            print('s.p', [s.P for s in self.strategiesOptionsInformation])
            print('s.F', [s.F for s in self.strategiesOptionsInformation])
            print('length', len([s for s in self.strategiesOptionsInformation if s.F > probability_threshold]))
            sys.exit(1)

    def make_deposit(self, amount):
        self.initialDeposit.amount = amount
        self.deposit = Deposit(amount, self.initialDeposit.lastPercentageWithdrawn)

    def withdraw_deposit(self, simulation=False):
        if self.isIntelligent:
            # Smart depositor
            bank_car = self.bank.get_capital_adequacy_ratio()
            shock = 0 if bank_car > self.safetyTreshold else ExogenousFactors.amountWithdrawn
        else:
            if simulation:
                # if in simulation, uses last real withdrawal by this depositor
                shock = self.deposit.lastPercentageWithdrawn
            else:
                # Simulating a Diamond & Dribvig banksim...
                shock = ExogenousFactors.amountWithdrawn if Util.get_random_uniform(
                    1) < ExogenousFactors.probabilityofWithdrawal else 0
        self.deposit.lastPercentageWithdrawn = shock
        amount_depositor_wish_to_withdraw = self.deposit.amount * shock
        amount_withdrawn = self.bank.withdraw_deposit(amount_depositor_wish_to_withdraw)
        self.deposit.amount -= amount_withdrawn
        self.amountEarlyWithdraw = amount_withdrawn

    def calculate_final_utility(self):
        if self.isIntelligent:
            strategy = self.currentlyChosenStrategy
            self.amountFinalWithdraw = self.deposit.amount
            final_consumption = self.amountEarlyWithdraw + self.amountFinalWithdraw

            if final_consumption < self.initialDeposit.amount:
                if ExogenousFactors.isDepositInsuranceAvailable:
                    final_consumption = self.initialDeposit.amount * (1 + ExogenousFactors.depositInterestRate)
                else:
                    strategy.insolvencyCounter += 1

            amount = self.initialDeposit.amount
            self.profit = 0
            if final_consumption != 0:
                self.profit = 100 * math.log(final_consumption / amount)
                profit = self.profit
            else:
                profit = self.profit
            strategy.finalConsumption = final_consumption
            strategy.strategyProfit = profit
            strategy.amountEarlyWithdraw = self.amountEarlyWithdraw
            strategy.amountFinalWithdraw = self.amountFinalWithdraw

    def reset(self):
        self.deposit = self.initialDeposit

    def period_0(self):
        if self.isIntelligent:
            self.update_strategy_choice_probability()
            self.pick_new_strategy()
            self.safetyTreshold = self.currentlyChosenStrategy.get_alpha_value()

    def period_1(self):
        #  Liquidity Shock
        if ExogenousFactors.areBankRunsPossible:
            self.withdraw_deposit()

    def period_2(self):
        pass


class Deposit:
    def __init__(self, amount=0, last_percentage_withdrawn=0):
        self.amount = amount
        self.lastPercentageWithdrawn = last_percentage_withdrawn


####################
####ACTIVATION######
####################

import itertools


class MultiStepActivation:

    def __init__(self, model):
        self.model = model
        self.cycle = 0
        self.period = 0

        # Agents
        self.central_bank = None
        self.clearing_house = None
        self.banks = []
        self.depositors = []
        self.LowRiskpoolcorporate_clients = []
        self.HighRiskpoolcorporate_clients = []

    def add_central_bank(self, central_bank):
        self.central_bank = central_bank

    def add_clearing_house(self, clearing_house):
        self.clearing_house = clearing_house

    def add_bank(self, bank):
        self.banks.append(bank)

    def add_depositor(self, depositor):
        self.depositors.append(depositor)

    def add_corporate_client_HighRisk(self, corporate_client):
        self.HighRiskpoolcorporate_clients.append(corporate_client)

    def add_corporate_client_LowRisk(self, corporate_client):
        self.LowRiskpoolcorporate_clients.append(corporate_client)

    @property
    def agents(self):
        # The order is important
        return itertools.chain(self.depositors, self.banks, [self.clearing_house], [self.central_bank],
                               self.HighRiskpoolcorporate_clients, self.LowRiskpoolcorporate_clients)

    def reset_cycle(self):
        self.cycle += 1
        for _ in self.agents:
            _.reset()

    def period_0(self):
        self.period = 0
        for _ in self.agents:
            _.period_0()

    def period_1(self):
        self.period = 1
        for _ in self.agents:
            _.period_1()

    def period_2(self):
        self.period = 2
        for _ in self.agents:
            _.period_2()


##############################
#######MODEL##################

from numba import prange

from mesa import Model


class BankingModel(Model):
    """
    BankSim is a banking agent-based simulation framework developed in Python 3+.

    Its main goal is to provide an out-of-the-box simulation tool to study the impacts of a broad range of regulation policies over the banking system.

    The basic model is based on the paper by Barroso, R. V. et al., Interbank network and regulation policies: an analysis through agent-based simulations with adaptive learning, published in the Journal Of Network Theory In Finance, v. 2, n. 4, p. 53–86, 2016.

    The paper is available online at https://mpra.ub.uni-muenchen.de/73308.
    """

    # ExpansiveMonetaryPolicy
    # RestrictiveMonetaryPolicy

    def __init__(self, simulation_type='General', exogenous_factors=None, number_of_banks=None):
        super().__init__()

        # Simulation data
        self.simulation_type = SimulationType[simulation_type]
        BankingModel.update_exogeneous_factors_by_simulation_type(self.simulation_type)

        BankingModel.update_exogeneous_factors(exogenous_factors, number_of_banks)

        # Economy data
        self.numberBanks = ExogenousFactors.numberBanks
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

        _params_corporate_clientsHighRisk = (ExogenousFactors.HighRiskCorporateClientDefaultRate,
                                             ExogenousFactors.HighRiskCorporateClientLossGivenDefault,
                                             ExogenousFactors.HighRiskCorporateClientLoanInterestRate)

        _params_corporate_clientsLowRisk = (ExogenousFactors.LowRiskCorporateClientDefaultRate,
                                            ExogenousFactors.LowRiskCorporateClientLossGivenDefault,
                                            ExogenousFactors.LowRiskCorporateClientLoanInterestRate)

        for bank in self.schedule.banks:
            for i in range(ExogenousFactors.numberDepositorsPerBank):
                depositor = Depositor(*_params_depositors, bank, self)
                bank.depositors.append(depositor)
                self.schedule.add_depositor(depositor)
            for i in range(ExogenousFactors.numberCorporateClientsPerBank):
                corporate_client = CorporateClient(*_params_corporate_clientsLowRisk, bank, self)
                bank.LowRiskpoolcorporateClients.append(corporate_client)
                self.schedule.add_corporate_client_LowRisk(corporate_client)
            for i in range(ExogenousFactors.numberCorporateClientsPerBank):
                corporate_client = CorporateClient(*_params_corporate_clientsHighRisk, bank, self)
                bank.HighRiskpoolcorporateClients.append(corporate_client)
                self.schedule.add_corporate_client_HighRisk(corporate_client)

    @jit(parallel=True)
    def step(self):
        self.schedule.reset_cycle()
        self.schedule.period_0()
        self.schedule.period_1()
        self.schedule.period_2()

    def run_model(self, n):
        for i in range(n):
            if (i % 10 == 0):
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
        if simulation_type == SimulationType.General:
            pass
        # elif simulation_type == SimulationType.RestrictiveMonetaryPolicy:
        #     ExogenousFactors.interbankInterestRate = 0.02
        #     ExogenousFactors.LowRiskCorporateClientDefaultRate = 0.06
        #     ExogenousFactors.HighRiskCorporateClientDefaultRate = 0.10
        #     ExogenousFactors.HighRiskCorporateClientLoanInterestRate = 0.12
        #     ExogenousFactors.LowRiskCorporateClientLoanInterestRate = 0.08
        #     ExogenousFactors.probabilityofWithdrawal = 0.25

        # elif simulation_type == SimulationType.ExpansiveMonetaryPolicy:
        #     ExogenousFactors.interbankInterestRate = 0.005
        #     ExogenousFactors.LowRiskCorporateClientDefaultRate = 0.02
        #     ExogenousFactors.HighRiskCorporateClientDefaultRate = 0.03
        #     ExogenousFactors.HighRiskCorporateClientLoanInterestRate = 0.10
        #     ExogenousFactors.LowRiskCorporateClientLoanInterestRate = 0.045
        #     ExogenousFactors.probabilityofWithdrawal = 0.15


from mesa.datacollection import DataCollector


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


########################
###
#####RUN THE MODEL ########
###########################
if __name__ == "__main__":

    number_repetitions = 1

    for i in range(number_repetitions):
        model = MyModel()
        model.run_model(50)

        ##########################
        ####PRINT THE OUTCOMES####
        ##########################

        # pd.set_option('max_columns', 100)

        ###############
        # Restrictive###
        ###############

        Modelo_original = model.datacollector.get_model_vars_dataframe()
        Modelo_original

        ##############
        # Expansive###
        ##############

        # Modelo_original2 = model.datacollector.get_model_vars_dataframe()
        # Modelo_original2

        ###############
        ####SAVING#### 

        ###################
        ####RESTRICTIVE####
        ###################

        Modelo_original.to_stata(r'output_Caso8_Desenv_Expansive' + str(i) + '.dta')

        ###################
        ####EXPANSIVE######
        ###################

        # Modelo_original2.to_stata(r'C:\Users\Douglas Silveira\Dropbox\PC\Documents\2021\JEDC - R1\Results\Expansionista.dta')
