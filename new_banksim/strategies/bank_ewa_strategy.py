class BankEWAStrategy:
    # capital ratio (capital / assets)
    numberAlphaOptions = 15
    alphaMaxValue = 1

    # liquidity ratio(liquid assets / deposits)
    numberBetaOptions = 15
    betaMaxValue = 1

    # Mark-up rate: real sector 1+i=c/1-p * (1+mu)
    numberMuROptions = 25
    MuRMaxValue = 1

    # Mark-up rate: interbank sector 1+i=c/1-p * (1+mu)
    numberMuIBOptions = 1
    MuIBMaxValue = 1

    def __init__(self, alpha_index_option=0, beta_index_option=0,
                 muR_index_option=0, muIB_index_option=0):
        self.alphaIndex = alpha_index_option
        self.betaIndex = beta_index_option
        self.MuRIndex = muR_index_option
        self.MuIBIndex = muIB_index_option
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0

    def get_alpha_value(self):
        a = self.alphaIndex/self.numberAlphaOptions
        return (self.alphaIndex + 1) / 100

    def get_beta_value(self):
        b = self.betaIndex / self.numberBetaOptions
        return (self.betaIndex + 1) / 100

    def get_MuR_value(self):
        m_r = self.MuRIndex / self.numberMuROptions
        return (self.MuRIndex + 1) / 100

    def get_MuIB_value(self):
        return (self.MuIBIndex + 1) / 100

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.alphaIndex == other.alphaIndex and self.betaIndex == other.betaIndex
        return False

    def reset(self):
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0

    @classmethod
    def bank_ewa_strategy_list(cls):
        return [BankEWAStrategy(a, b, c, d) for a in range(cls.numberAlphaOptions) for b in range(cls.numberBetaOptions) \
                for c in range(cls.numberMuROptions) for d in range(cls.numberMuIBOptions)]
