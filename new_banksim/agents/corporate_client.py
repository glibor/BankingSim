from mesa import Agent

from util import Util


class CorporateClient(Agent):

    def __init__(self, default_rate, loss_given_default, loan_interest_rate, bank, model):
        super().__init__(Util.get_unique_id(), model)

        # Bank Reference
        # self.bank = bank

        self.loanAmount = 0
        self.percentageRepaid = 0

        self.probabilityOfDefault = default_rate
        self.lossGivenDefault = loss_given_default
        self.loanInterestRate = loan_interest_rate
        self.loanDemandSize = 1.5/200
        self.creditElasticity = 10
        self.realSectorHelper = RealSectorHelper(self)
        self.creditDemandFulfilled = False

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

    def credit_demand(self, interest_rate):
        loan = self.loanDemandSize * interest_rate ** -self.creditElasticity
        return loan

    def reset(self):
        self.loanAmount = 0
        self.percentageRepaid = 0
        self.creditDemandFulfilled = False
        self.realSectorHelper.alreadyBorrowed = self.realSectorHelper.amountLiquidityLeftToBorrowOrLend = 0

    def period_0(self):
        pass

    def period_1(self):
        pass

    def period_2(self):
        pass


class RealSectorHelper:

    def __init__(self, firm):
        self.firm = firm
        self.amountLiquidityLeftToBorrowOrLend = 0
        self.alreadyBorrowed = 0

    def calculate_credit_demand(self, interest_rate, already_borrowed):
        amount = max(0,self.firm.credit_demand(interest_rate) - already_borrowed)
        return amount

    def update_amount(self, interest_rate):
        amount = self.calculate_credit_demand(interest_rate, self.alreadyBorrowed)
        self.amountLiquidityLeftToBorrowOrLend = -amount

    def get_loan(self, amount):
        self.amountLiquidityLeftToBorrowOrLend += amount
        self.alreadyBorrowed += amount
