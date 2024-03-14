from copy import copy
import time
import numpy as np
# from numba import njit
from mesa import Agent
from numpy.random import choice
from random import choices
import warnings

from exogeneous_factors import BankSizeDistribution, ExogenousFactors
from strategies.bank_ewa_strategy import BankEWAStrategy
from util import Util


class Bank(Agent):

    def __init__(self, bank_size_distribution, is_intelligent, ewa_damping_factor, model, bank_id):
        super().__init__(Util.get_unique_id(), model)

        self.initialSize = 1 if bank_size_distribution != BankSizeDistribution.LogNormal \
            else Util.get_random_log_normal(-0.5, 1)

        self.is_bank = True
        self.bank_id = bank_id
        self.interbankHelper = InterbankHelper()
        self.guaranteeHelper = GuaranteeHelper()
        self.realSectorHelper = RealSectorHelper()
        self.depositors = []  # Depositors
        self.corporateClients = []  # CorporateClients
        self.creditSupplyExhausted = False

        self.liquidityNeeds = 0
        self.bankRunOccurred = False
        self.withdrawalsCounter = 0

        self.balanceSheet = BalanceSheet()
        self.auxBalanceSheet = None
        self.lastProfit = 0

        self.isIntelligent = is_intelligent
        if self.isIntelligent:
            # self.strategiesOptionsInformation = BankEWAStrategy.bank_ewa_strategy_list()
            self.StrategiesId = self.model.bankStrategiesId
            self.A, self.P, self.strategyProfitPercentageDamped = 3 * [np.zeros(self.model.bankStrategiesLength)]
            self.currentlyChosenStrategy = None
            self.currentlyChosenStrategyId = None
            self.EWADampingFactor = ewa_damping_factor

    def update_strategy_choice_probability(self):
        p = np.max(self.P)
        if p < 0.999:
            self.A = self.A + 2 * self.strategyProfitPercentageDamped# / self.model.current_step  # 0.9999 *
            m = np.max(self.A)
            #if p > .9:
            #    print(p)
             #   print(m,np.max(self.strategyProfitPercentageDamped))
            self.P = np.exp(self.A) / (np.sum(np.exp(self.A)))

            if m > 1000:
                self.limit_array()
                # print(np.max(self.A))
        #else:
        #    print(p)
            #print(np.max(self.strategyProfitPercentageDamped))
        #elif p > 1:
        #    self.P = self.P / p

    def limit_array(self):
        m = np.max(self.A)
        new = self.A + 100 - m
        self.A = new
        self.P = np.exp(self.A) / (np.sum(np.exp(self.A)))

        # print('Rolor',m)

    def pick_new_strategy(self):
        probability_threshold = Util.get_random_uniform(1)
        # _id = np.random.choice(self.StrategiesId, p=self.P)  # choice(self.StrategiesId, p=self.P)
        # _id = np.random.choice(self.StrategiesId, p=self.P)  # choice(self.StrategiesId, p=self.P)
        try:
            _id = np.random.choice(self.StrategiesId, p=self.P)  # choice(self.StrategiesId, p=self.P)
        except ValueError:
            a = self.A
            tmp = np.where(a > 0, a, 0)
            tmp = np.where(tmp < 10, a, 10)
            print(tmp)
            print(np.max(tmp))
            print(np.sum(tmp), 'cool')
            # self.P = tmp / np.sum(tmp)
            self.A = tmp
            self.P = np.exp(tmp) / (np.sum(np.exp(tmp)))
            print(self.P)
            _id = choice(self.StrategiesId, p=self.P)
        self.currentlyChosenStrategyId = _id
        self.currentlyChosenStrategy = self.model.bankStrategies[_id]

    @staticmethod
    # @njit
    def choice(strats, p):
        random = np.random.random()
        count = 0
        while random > p[count]:
            count += 1
        return strats[count]

    def reset(self):
        self.liquidityNeeds = 0
        self.bankRunOccurred = False
        self.withdrawalsCounter = 0
        self.creditSupplyExhausted = False

    def reset_collateral(self):
        self.guaranteeHelper = GuaranteeHelper()

    def setup_balance_sheet_intelligent(self, strategy=None):  # TODO: ATUALIZAR PARA A NOVA VERSãO
        if strategy is None:
            strategy = self.currentlyChosenStrategy
        risk_appetite = strategy.get_gamma_value()

        self.balanceSheet.liquidAssets = self.initialSize * strategy.get_beta_value()
        self.balanceSheet.highRiskLoans = (self.initialSize - self.balanceSheet.liquidAssets) * risk_appetite
        self.balanceSheet.nonFinancialSectorLoan = self.initialSize - self.balanceSheet.liquidAssets - self.balanceSheet.highRiskLoans
        self.balanceSheet.interbankLoan = 0
        self.balanceSheet.discountWindowLoan = 0
        self.balanceSheet.deposits = self.initialSize * (strategy.get_alpha_value() - 1)
        self.liquidityNeeds = 0
        self.setup_balance_sheet()

    def setup_balance_sheet(self):
        # loan_per_coporate_client = self.balanceSheet.nonFinancialSectorLoan / len(self.corporateClients)
        # for corporateClient in self.corporateClients:
        # corporateClient.loanAmount = loan_per_coporate_client

        deposit_per_depositor = -self.balanceSheet.deposits / len(self.depositors)
        for depositor in self.depositors:
            depositor.make_deposit(deposit_per_depositor)

    def get_capital_adequacy_ratio(self):
        if self.is_solvent():
            rwa = self.get_real_sector_risk_weighted_assets()
            total_risk_weighted_assets = self.balanceSheet.liquidAssets * ExogenousFactors.CashRiskWeight + rwa

            if self.is_interbank_creditor():
                total_risk_weighted_assets += self.balanceSheet.interbankLoan * ExogenousFactors.InterbankLoanRiskWeight

            if total_risk_weighted_assets != 0:
                return -self.balanceSheet.capital / total_risk_weighted_assets
        return 0

    def adjust_capital_ratio(self, minimum_capital_ratio_required):
        current_capital_ratio = self.get_capital_adequacy_ratio()

        real_sector_clearing_house = self.model.schedule.real_sector_clearing_house
        lender_id = self.bank_id
        lending_vector = real_sector_clearing_house.realSectorLendingMatrix[lender_id, :]
        high_risk_lending_vector = real_sector_clearing_house.highRiskLendingMatrix[lender_id, :]
        if current_capital_ratio <= minimum_capital_ratio_required:
            adjustment_factor = current_capital_ratio / minimum_capital_ratio_required

            self.model.schedule.real_sector_clearing_house.realSectorLendingMatrix[lender_id,
            :] = lending_vector * adjustment_factor
            self.model.schedule.real_sector_clearing_house.highRiskLendingMatrix[lender_id,
            :] = high_risk_lending_vector * adjustment_factor
            self.update_non_financial_sector_loans(lending_vector, high_risk_lending_vector)

    def adjust_capital_ratio_alternative(self, minimum_capital_ratio_required):
        current_capital_ratio = self.get_capital_adequacy_ratio()
        print(self.balanceSheet.liquidAssets, 'cool')
        if current_capital_ratio <= minimum_capital_ratio_required:
            adjustment_factor = current_capital_ratio / minimum_capital_ratio_required
            initial_hr = self.balanceSheet.highRiskLoans
            initial = self.balanceSheet.nonFinancialSectorLoan
            self.balanceSheet.highRiskLoans = self.balanceSheet.highRiskLoans * adjustment_factor
            self.balanceSheet.nonFinancialSectorLoan = self.balanceSheet.nonFinancialSectorLoan * adjustment_factor
            self.balanceSheet.liquidAssets += initial - self.balanceSheet.nonFinancialSectorLoan + initial_hr - self.balanceSheet.highRiskLoans
        print(self.balanceSheet.liquidAssets)

    def update_non_financial_sector_loans(self, lending_vector, high_risk_lending_vector):
        # print(self.bank_id)
        # print(self.balanceSheet.nonFinancialSectorLoan-np.sum(lending_vector))
        self.balanceSheet.liquidAssets += self.balanceSheet.nonFinancialSectorLoan - np.sum(
            lending_vector) + self.balanceSheet.highRiskLoans - np.sum(high_risk_lending_vector)
        self.balanceSheet.nonFinancialSectorLoan = np.sum(lending_vector)
        self.balanceSheet.highRiskLoans = np.sum(high_risk_lending_vector)

    def get_real_sector_risk_weighted_assets(self):
        """if ExogenousFactors.standardCorporateClients:
                    pass
                else:
                    real_sector_clearing_house = self.model.schedule.real_sector_clearing_house
                    lender_id = self.bank_id
                    lending_vector = real_sector_clearing_house.realSectorLendingMatrix[lender_id, :]
                    risk_weighted = np.sum(lending_vector) * ExogenousFactors.CorporateLoanRiskWeight
                    lending_vector = real_sector_clearing_house.highRiskLendingMatrix[lender_id, :]
                    risk_weighted += lending_vector * ExogenousFactors.HighRiskCorporateLoanRiskWeight
                    return risk_weighted
        """
        return (self.balanceSheet.nonFinancialSectorLoan * ExogenousFactors.CorporateLoanRiskWeight +
                self.balanceSheet.highRiskLoans * ExogenousFactors.HighRiskCorporateLoanRiskWeight)

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

    def collect_loans(self, real_sector_clearing_house):

        # Standard loans
        lender_id = self.bank_id
        lending_vector = real_sector_clearing_house.realSectorLendingMatrix[lender_id, :]
        interest_rate_vector = real_sector_clearing_house.realSectorInterestMatrix[lender_id, :]
        LGD_vector = real_sector_clearing_house.defaultLGDMatrix
        prob_matrix = real_sector_clearing_house.defaultProbabilityMatrix

        initial = np.sum(real_sector_clearing_house.realSectorLendingMatrix[lender_id, :])
        for n, i in enumerate(lending_vector):
            rand = Util.get_random_uniform(1)
            if rand <= prob_matrix[0, n]:
                amount_paid = i * (1 - LGD_vector[0, n])

            else:
                amount_paid = i * (1 + interest_rate_vector[n])
            lending_vector[n] = amount_paid
        # print(100*'-')
        # print("Initial: "+str(np.sum(initial)))
        # print('Gain: '+str(np.sum(lending_vector)-initial))
        # print(100 * '-')
        self.balanceSheet.nonFinancialSectorLoan = np.sum(lending_vector)

        # High risk loans

        lending_vector = real_sector_clearing_house.highRiskLendingMatrix[lender_id, :]
        interest_rate_vector = real_sector_clearing_house.highRiskInterestMatrix[lender_id, :]

        LGD_vector = real_sector_clearing_house.defaultLGDMatrix
        prob_matrix = real_sector_clearing_house.highRiskDefaultProbabilityMatrix

        for n, i in enumerate(lending_vector):
            rand = Util.get_random_uniform(1)

            if rand <= prob_matrix[0, n]:

                amount_paid = i * (1 - LGD_vector[0, n])
            else:
                amount_paid = i * (1 + interest_rate_vector[n])

            lending_vector[n] = amount_paid

        self.balanceSheet.highRiskLoans = np.sum(lending_vector)

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
            liquidity_needed = -self.liquidityNeeds
            total_loans_to_sell = liquidity_needed * (1 + ExogenousFactors.illiquidAssetDiscountRate)
            if self.balanceSheet.nonFinancialSectorLoan > total_loans_to_sell:
                amount_sold = total_loans_to_sell
                self.liquidityNeeds = 0
                self.balanceSheet.deposits += liquidity_needed
            else:
                amount_sold = self.balanceSheet.nonFinancialSectorLoan
                self.liquidityNeeds += amount_sold / (1 + ExogenousFactors.illiquidAssetDiscountRate)
                self.balanceSheet.deposits += liquidity_needed - self.liquidityNeeds
            proportion_of_illiquid_assets_sold = amount_sold / self.balanceSheet.nonFinancialSectorLoan

            for firm in self.corporateClients:
                firm.loanAmount *= 1 - proportion_of_illiquid_assets_sold
            self.balanceSheet.nonFinancialSectorLoan -= amount_sold

    def get_profit(self):
        resulting_capital = self.balanceSheet.assets + self.balanceSheet.liabilities
        original_capital = self.auxBalanceSheet.assets + self.auxBalanceSheet.liabilities
        if ExogenousFactors.banksHaveLimitedLiability:
            resulting_capital = max(resulting_capital, 0)
        return resulting_capital - original_capital

    def calculate_profit(self, minimum_capital_ratio_required):
        if self.isIntelligent:
            strategyId = self.currentlyChosenStrategyId

            self.bankRunOccurred = (self.withdrawalsCounter > ExogenousFactors.numberDepositorsPerBank / 2)
            # print(self.withdrawalsCounter,ExogenousFactors.numberDepositorsPerBank / 2)
            if self.bankRunOccurred:
                original_loans = self.auxBalanceSheet.totalNonFinancialLoans
                resulting_loans = self.balanceSheet.totalNonFinancialLoans
                delta = original_loans - resulting_loans
                if delta > 0:
                    self.balanceSheet.nonFinancialSectorLoan -= delta * 0.02
                    self.balanceSheet.highRiskLoans -= delta * 0.02

            profit = self.get_profit()
            # strategy.strategyProfit = profit

            if ExogenousFactors.isCapitalRequirementActive:
                current_capital_ratio = self.get_capital_adequacy_ratio()

                if current_capital_ratio < minimum_capital_ratio_required:
                    delta_capital_ratio = minimum_capital_ratio_required - current_capital_ratio
                    profit -= delta_capital_ratio

            # Return on Equity, based on initial shareholders equity.

            strategyProfitPercentage = -(profit / self.auxBalanceSheet.capital)
            # strategyProfitPercentage=profit
            self.lastProfit = strategyProfitPercentage
            self.strategyProfitPercentageDamped = np.zeros(self.model.bankStrategiesLength)
            self.strategyProfitPercentageDamped[strategyId] = strategyProfitPercentage * self.EWADampingFactor

    def liquidate(self):
        #  first, sell assets...
        self.balanceSheet.liquidAssets += self.balanceSheet.totalNonFinancialLoans
        self.balanceSheet.nonFinancialSectorLoan = self.balanceSheet.highRiskLoans = 0

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

        percentage_deposits_payable = self.balanceSheet.liquidAssets / np.absolute(self.balanceSheet.deposits)

        self.balanceSheet.deposits *= percentage_deposits_payable

        for depositor in self.depositors:
            depositor.deposit.amount *= percentage_deposits_payable

        self.balanceSheet.liquidAssets = 0

    def get_wacc(self):
        strategy = self.currentlyChosenStrategy
        c = (1 + self.model.depositInterestRate) * (1 - strategy.get_alpha_value()) + strategy.get_alpha_value()

        return c

    def get_real_sector_interest_rate(self, default_probability):
        strategy = self.currentlyChosenStrategy
        mu = strategy.get_MuR_value()
        exo_rate = ExogenousFactors.liquidAssetsInterestRate
        p = default_probability
        c = self.get_wacc()
        i = max(max((c / (1 - p)) * (1 + mu), 1) - 1, exo_rate)  # max(max(c * (1 + mu), 1) - 1, exo_rate)
        # print(c, i,p)

        return i

    def get_interbank_interest_rate(self):
        strategy = self.currentlyChosenStrategy
        exo_rate = ExogenousFactors.liquidAssetsInterestRate
        mu = strategy.get_MuIB_value()
        c = self.get_wacc()
        i = max(max(c * (1 + mu), 1) - 1, exo_rate)

        return i

    @property
    def probability_entropy(self):
        p = self.P
        lp = np.log(self.P)
        return -np.sum(p*lp)

    @property
    def portfolio_risk(self):
        real_sector_clearing_house = self.model.schedule.real_sector_clearing_house
        lender_id = self.bank_id
        lending_vector = real_sector_clearing_house.realSectorLendingMatrix[lender_id, :]
        risk = real_sector_clearing_house.defaultProbabilityMatrix
        risk_weights = np.multiply(lending_vector, risk)
        return np.sum(risk_weights / np.sum(lending_vector))

    def is_insolvent(self):
        return self.balanceSheet.capital >= 0

    def is_solvent(self):
        return self.balanceSheet.capital < 0

    def is_interbank_creditor(self):
        return self.balanceSheet.interbankLoan >= 0

    def is_interbank_debtor(self):
        return self.balanceSheet.interbankLoan < 0

    def period_0(self):

        if self.isIntelligent:
            self.update_strategy_choice_probability()
            self.pick_new_strategy()
            self.setup_balance_sheet_intelligent(self.currentlyChosenStrategy)
            if ExogenousFactors.isCapitalRequirementActive:
                pass  # self.adjust_capital_ratio_alternative(ExogenousFactors.minimumCapitalAdequacyRatio)


        else:
            self.setup_balance_sheet()
        # print('Period 0')
        # print(self.balanceSheet)

        # self.auxBalanceSheet = copy(self.balanceSheet)

    def period_1(self):
        # print(self.balanceSheet.nonFinancialSectorLoan)
        # print(self.balanceSheet.assets+self.balanceSheet.liabilities)
        # First, banks try to use liquid assets to pay early withdrawals...
        self.use_liquid_assets_to_pay_depositors_back()

        # ... if needed, they will try interbank market by clearing house.
        # ... if banks still needs liquidity, central bank might rescue...

    def period_2(self):
        resulting_capital = self.balanceSheet.assets + self.balanceSheet.liabilities
        original_capital = self.auxBalanceSheet.assets + self.auxBalanceSheet.liabilities
        # print('Period 2.1')

        # print("+"*100)
        # print(self.balanceSheet.capital)
        self.accrue_interest_balance_sheet()
        self.collect_loans(self.model.schedule.real_sector_clearing_house)
        if not self.is_liquid():
            print('top', self.balanceSheet)


class InterbankHelper:
    def __init__(self):
        self.counterpartyID = 0
        self.priorityOrder = 0
        self.auxPriorityOrder = 0
        self.loanAmount = 0
        self.acumulatedLiquidity = 0
        self.riskSorting = None
        self.amountLiquidityLeftToBorrowOrLend = 0


class RealSectorHelper:
    def __init__(self):
        self.counterpartyID = 0
        self.priorityOrder = 0
        self.auxPriorityOrder = 0
        self.loanAmount = 0
        self.acumulatedLiquidity = 0
        self.riskSorting = None
        self.amountLiquidityLeftToBorrowOrLend = 0
        self.amountLiquidityLeftToBorrowOrLendHighRisk = 0


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
        self.nonFinancialSectorLoan = 0
        self.highRiskLoans = 0
        self.liquidAssets = 0

    @property
    def capital(self):
        return -(self.liquidAssets +
                 self.totalNonFinancialLoans +
                 self.interbankLoan +
                 self.discountWindowLoan +
                 self.deposits)

    @property
    def assets(self):
        return self.liquidAssets + self.totalNonFinancialLoans + np.max(self.interbankLoan, 0)

    @property
    def liabilities(self):
        return self.deposits + self.discountWindowLoan + np.min(self.interbankLoan, 0)

    @property
    def totalNonFinancialLoans(self):
        return self.highRiskLoans + self.nonFinancialSectorLoan

    def __repr__(self):
        name = """
Assets

Liquid assets: {}
Non financial loans: {}
High risk loans: {}
Interbank: {}

Liabilities

Capital: {}
Deposits: {}
Interbank: {}
Discount window: {}
        """
        lista = [self.liquidAssets, self.nonFinancialSectorLoan, self.highRiskLoans, max(self.interbankLoan, 0),
                 self.capital, self.deposits, min(self.interbankLoan, 0), self.discountWindowLoan]
        final = name.format(*lista)
        return final
