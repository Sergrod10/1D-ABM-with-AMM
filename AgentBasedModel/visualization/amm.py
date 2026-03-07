from AgentBasedModel.simulator import AmmAgentInfo
import AgentBasedModel.utils.math as math
import matplotlib.pyplot as plt


def plot_price_amm(info: AmmAgentInfo, rolling: int = 1, figsize=(6, 6)):
    """
    Plot AMM spot price over iterations.
    """
    plt.figure(figsize=figsize)
    plt.title('AMM Price') if rolling == 1 else plt.title(f'AMM Price (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    plt.plot(range(rolling - 1, len(info.prices)), math.rolling(info.prices, rolling), color='black')
    plt.show()

def plot_cash_amm(info: AmmAgentInfo, rolling: int = 1, figsize=(6, 6)):
    """
    Plot AMM cash reserve dynamics over iterations.
    """
    plt.figure(figsize=figsize)
    plt.title('AMM cash') if rolling == 1 else plt.title(f'AMM cash (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('cash')
    plt.plot(range(rolling - 1, len(info.cash_amm)), math.rolling(info.cash_amm, rolling), color='black')
    plt.show()

def plot_assets_amm(info: AmmAgentInfo, rolling: int = 1, figsize=(6, 6)):
    """
    Plot AMM asset reserve dynamics over iterations.
    """
    plt.figure(figsize=figsize)
    plt.title('AMM assets') if rolling == 1 else plt.title(f'AMM assets (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('assets')
    plt.plot(range(rolling - 1, len(info.assets_amm)), math.rolling(info.assets_amm, rolling), color='black')
    plt.show()

def plot_equities_amm(info: AmmAgentInfo, rolling: int = 1, figsize=(6, 6)):
    """
    Plot AMM total equity (cash + assets valued by AMM price).
    """
    plt.figure(figsize=figsize)
    plt.title('AMM equities') if rolling == 1 else plt.title(f'AMM equities (MA {rolling})')
    plt.xlabel('Iterations')
    plt.ylabel('equities')
    plt.plot(range(rolling - 1, len(info.equities_amm)), math.rolling(info.equities_amm, rolling), color='black')
    plt.show()
