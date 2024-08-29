# <summary>
# continuous model for discounting a coupon bond
# </summary>
import numpy as np

class couponBond:
    def __init__(self, principal, rate, maturity, market_rate):
        self.principal = principal
        self.rate = rate/100
        self.maturity = maturity
        self.market_rate = market_rate/100
    # calculate present value from a principal value
    def present_value(self, x, n):
        return x*np.exp(-self.market_rate*n)
    # calculate price of a bond using continuous model
    def calculate_price(self):
        price = 0
        for i in range(1, self.maturity+1):
            price = price + self.present_value((self.principal*self.rate), i)

        price = price + self.present_value(self.principal, self.maturity)
        return price
# allow user to set parameter for calculation
if __name__ == '__main__':
    principal = int(input("Enter principal value:"))
    rate = int(input("Enter coupon bond rate as %:"))
    maturity = int(input("Enter bond maturity:"))
    market_rate = int(input("Lastly, enter market interest rate as %:"))
    bond = couponBond(principal, rate, maturity, market_rate)
    print("Bond price is: %.2f" % bond.calculate_price())