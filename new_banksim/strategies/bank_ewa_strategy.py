class BankEWAStrategy:
    # capital ratio (capital / liabilities)
    numberAlphaOptions = 10#30#15
    alphaMaxValue = .3
    alphaMinValue = .05

    # liquidity ratio(liquid assets / total assets)
    numberBetaOptions = 10
    betaMaxValue = 0.4


    # risk appetite (High Risk Corporate Client/ Low Risk Corporate Client)
    numberGammaOptions = 10
    gammaMaxValue = 0.3

    # Mark-up rate: real sector 1+i=c/1-p * (1+mu)
    numberMuROptions = 10#30#15
    MuRMaxValue = .25

    # Mark-up rate: interbank sector 1+i=c/1-p * (1+mu)
    numberMuIBOptions = 10#30#15
    MuIBMaxValue = .25 #.5

    def __init__(self, alpha_index_option=0, beta_index_option=0,gamma_index_option=0,
                 muR_index_option=0, muIB_index_option=0):
        self.alphaIndex = alpha_index_option
        self.betaIndex = beta_index_option
        self.gammaIndex = gamma_index_option

        self.MuRIndex = muR_index_option
        self.MuIBIndex = muIB_index_option

        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0

    def get_alpha_value(self):
        a = (self.alphaIndex + 1) / (self.numberAlphaOptions)
        return a * self.alphaMaxValue + self.alphaMinValue # (self.alphaIndex + 1) / 100

    def get_beta_value(self):
        b = (self.betaIndex + 1) / (self.numberBetaOptions)
        return b * self.betaMaxValue  # (self.betaIndex + 1) / 100

    def get_gamma_value(self):
        g = (self.gammaIndex + 1) / (self.numberGammaOptions)
        return g * self.gammaMaxValue  # (self.betaIndex + 1) / 100

    def get_MuR_value(self):
        m_r = (self.MuRIndex + 1) / (self.numberMuROptions)
        return m_r * self.MuRMaxValue  # (self.MuRIndex + 1) / 100

    def get_MuIB_value(self):
        m_ib = (self.MuIBIndex + 1) / (self.numberMuIBOptions)
        return m_ib * self.MuIBMaxValue  # a * self.alphaMaxValue#(self.MuIBIndex + 1) / 100

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.alphaIndex == other.alphaIndex and self.betaIndex == other.betaIndex
        return False

    def reset(self):
        self.strategyProfit = self.strategyProfitPercentage = self.strategyProfitPercentageDamped = 0
        self.A = self.P = self.F = 0

    @classmethod
    def bank_ewa_strategy_list(cls):
        return [BankEWAStrategy(a, b, c, d, e) for a in range(cls.numberAlphaOptions) for b in range(cls.numberBetaOptions) \
                for c in range(cls.numberGammaOptions) for d in range(cls.numberMuROptions) for e in range(cls.numberMuIBOptions)]

    def __str__(self):
        name = "Alpha:"+str(self.get_alpha_value())+" Beta:"+str(self.get_beta_value())+" MuR:"+str(self.get_MuR_value())
        return name

