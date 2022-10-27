from copy import copy
import time
import numpy as np
from numba import njit
from mesa import Agent
from numpy.random import choice
from random import choices

from exogeneous_factors import BankSizeDistribution, ExogenousFactors
from strategies.bank_ewa_strategy import BankEWAStrategy
from util import Util


class Bank(Agent):

    def __init__(self, bank_size_distribution, is_intelligent, ewa_damping_factor, model):
        super().__init__(Util.get_unique_id(), model)

        self.initialSize = 1 if bank_size_distribution != BankSizeDistribution.LogNormal \
            else Util.get_random_log_normal(-0.5, 1)

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

        self.isIntelligent = is_intelligent
        if self.isIntelligent:
            self.strategiesOptionsInformation = BankEWAStrategy.bank_ewa_strategy_list()
            self.currentlyChosenStrategy = None
            self.EWADampingFactor = ewa_damping_factor

    def update_strategy_choice_probability(self):
        list_A, list_damp_pct = np.array([s.A for s in self.strategiesOptionsInformation]), np.array(
            [s.strategyProfitPercentageDamped for s in self.strategiesOptionsInformation])
        list_a = self.update_helper(list_A, list_damp_pct)
        _exp = np.exp(list_a)
        list_p = _exp / (np.sum(_exp))

        list_f = np.cumsum(list_p)
        for i, strategy in enumerate(self.strategiesOptionsInformation):
            strategy.A, strategy.P, strategy.F = list_a[i], list_p[i], list_f[i]

    @staticmethod
    @njit
    def update_helper(l1, l2):
        return 0.9999 * l1 + 0.005 * l2

    def pick_new_strategy(self):
        probability_threshold = Util.get_random_uniform(1)
        self.currentlyChosenStrategy = [s for s in self.strategiesOptionsInformation if s.F > probability_threshold][0]

    @staticmethod
    @njit
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
        self.balanceSheet.liquidAssets = self.initialSize * strategy.get_beta_value()
        self.balanceSheet.nonFinancialSectorLoan = self.initialSize - self.balanceSheet.liquidAssets
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

    def adjust_capital_ratio(self, minimum_capital_ratio_required):  # TODO: Ajustar para novo modelo de clientes
        current_capital_ratio = self.get_capital_adequacy_ratio()
        if current_capital_ratio <= minimum_capital_ratio_required:
            adjustment_factor = current_capital_ratio / minimum_capital_ratio_required
            for corporateClient in self.corporateClients:
                original_loan_amount = corporateClient.loanAmount
                new_loan_amount = original_loan_amount * adjustment_factor
                corporateClient.loanAmount = new_loan_amount
                self.balanceSheet.liquidAssets += (original_loan_amount - new_loan_amount)

            # self.update_non_financial_sector_loans()

    def update_non_financial_sector_loans(self):
        self.balanceSheet.nonFinancialSectorLoan = sum(
            client.loanAmount for client in self.corporateClients)

    def get_real_sector_risk_weighted_assets(self):
        if ExogenousFactors.standardCorporateClients:
            return self.balanceSheet.nonFinancialSectorLoan * ExogenousFactors.CorporateLoanRiskWeight
        else:
            for corporateClient in self.corporateClients:
                if corporateClient.probabilityOfDefault == ExogenousFactors.retailCorporateClientDefaultRate:
                    return corporateClient.loanAmount * ExogenousFactors.retailCorporateLoanRiskWeight
                elif corporateClient.probabilityOfDefault == ExogenousFactors.wholesaleCorporateClientDefaultRate:
                    return corporateClient.loanAmount * ExogenousFactors.wholesaleCorporateLoanRiskWeight
                else:
                    # default risk weight
                    return corporateClient.loanAmount * ExogenousFactors.CorporateLoanRiskWeight

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

    def collect_loans(self, real_sector_clearing_house):  # TODO: UPDATE BASED IN THE REAL SECTOR CLEARING HOUSE MATRIX
        lender_id = (self.unique_id - real_sector_clearing_house.numberBanks) % real_sector_clearing_house.numberBanks
        lending_vector = real_sector_clearing_house.realSectorLendingMatrix[lender_id, :]
        interest_rate_vector = real_sector_clearing_house.realSectorInterestMatrix[lender_id, :]
        LGD_vector = real_sector_clearing_house.defaultLGDMatrix
        prob_matrix = real_sector_clearing_house.defaultProbabilityMatrix

        # print("Antes",sum(lending_vector))

        for n, i in enumerate(lending_vector):

            if Util.get_random_uniform(1) <= prob_matrix[0, n]:
                amount_paid = i * (1 - LGD_vector[0, n])
            else:
                amount_paid = i * (1 + interest_rate_vector[n])
            lending_vector[n] = amount_paid

        self.balanceSheet.nonFinancialSectorLoan = sum(lending_vector)

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
        # print(resulting_capital,original_capital)
        return resulting_capital - original_capital

    def calculate_profit(self, minimum_capital_ratio_required):
        if self.isIntelligent:
            strategy = self.currentlyChosenStrategy

            self.bankRunOccurred = (self.withdrawalsCounter > ExogenousFactors.numberDepositorsPerBank / 2)
            # print(self.withdrawalsCounter,ExogenousFactors.numberDepositorsPerBank / 2)
            if self.bankRunOccurred:
                original_loans = self.auxBalanceSheet.nonFinancialSectorLoan
                resulting_loans = self.balanceSheet.nonFinancialSectorLoan
                delta = original_loans - resulting_loans
                if delta > 0:
                    self.balanceSheet.nonFinancialSectorLoan -= delta * 0.02

            profit = self.get_profit()

            strategy.strategyProfit = profit

            if ExogenousFactors.isCapitalRequirementActive:
                current_capital_ratio = self.get_capital_adequacy_ratio()

                if current_capital_ratio < minimum_capital_ratio_required:
                    delta_capital_ratio = minimum_capital_ratio_required - current_capital_ratio
                    strategy.strategyProfit -= delta_capital_ratio

            # Return on Equity, based on initial shareholders equity.
            strategy.strategyProfitPercentage = -strategy.strategyProfit / self.auxBalanceSheet.capital
            strategy.strategyProfitPercentageDamped = strategy.strategyProfitPercentage * self.EWADampingFactor

    def liquidate(self):
        #  first, sell assets...
        self.balanceSheet.liquidAssets += self.balanceSheet.nonFinancialSectorLoan
        self.balanceSheet.nonFinancialSectorLoan = 0

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
        c = (1 + self.model.depositInterestRate) * (1 - strategy.get_alpha_value())
        return c

    def get_real_sector_interest_rate(self, default_probability):
        strategy = self.currentlyChosenStrategy
        mu = strategy.get_MuR_value()
        p = default_probability
        c = self.get_wacc()
        # print((c / (1 - p))*(1/(1+1)))
        i = max((c / (1 - p)) * (1 + mu), 1) - 1

        return i

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
            self.setup_balance_sheet_intelligent(self.currentlyChosenStrategy)
        else:
            self.setup_balance_sheet()
        # self.auxBalanceSheet = copy(self.balanceSheet)

    def period_1(self):
        # First, banks try to use liquid assets to pay early withdrawals...
        self.use_liquid_assets_to_pay_depositors_back()
        # ... if needed, they will try interbank market by clearing house.
        # ... if banks still needs liquidity, central bank might rescue...

    def period_2(self):
        self.accrue_interest_balance_sheet()
        self.collect_loans(self.model.schedule.real_sector_clearing_house)


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
        self.liquidAssets = 0

    @property
    def capital(self):
        return -(self.liquidAssets +
                 self.nonFinancialSectorLoan +
                 self.interbankLoan +
                 self.discountWindowLoan +
                 self.deposits)

    @property
    def assets(self):
        return self.liquidAssets + self.nonFinancialSectorLoan + np.max(self.interbankLoan, 0)

    @property
    def liabilities(self):
        return self.deposits + self.discountWindowLoan + np.min(self.interbankLoan, 0)
