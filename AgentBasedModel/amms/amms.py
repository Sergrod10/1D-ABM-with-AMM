from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from AgentBasedModel.agents import Arbitrage


class ArbitrageOrder:
    """
    ArbitrageOrder stores one arbitrage request that will be processed in AMM queue.
    """
    order_id = 0

    def __init__(self, qty, min_got_cash, pay_for_prio, order_type, trader_link: Optional["Arbitrage"] = None):
        """
        :param qty: quantity to arbitrage
        :param min_got_cash: minimal acceptable cash output of the swap
        :param pay_for_prio: payment used for queue priority
        :param order_type: direction of arbitrage, 'buy' or 'sell'
        :param trader_link: trader that submitted this order
        """
        # Properties
        self.qty = qty
        self.min_got_cash = min_got_cash
        self.pay_for_prio = pay_for_prio
        self.order_type = order_type
        self.trader = trader_link
        self.order_id = ArbitrageOrder.order_id

        ArbitrageOrder.order_id += 1

    def __lt__(self, other) -> bool:
        """
        Sort by priority payment, then by guaranteed output, then by order creation time.
        """
        if self.pay_for_prio != other.pay_for_prio:
            return self.pay_for_prio > other.pay_for_prio
        if self.min_got_cash != other.min_got_cash:
            return self.min_got_cash > other.min_got_cash
        return self.order_id < other.order_id

    def to_dict(self) -> dict:
        """
        :return: transformation from order object into dict object
        """
        return {'qty': self.qty, 'min_got_cash': self.min_got_cash, 'pay_for_prio': self.pay_for_prio, 'order_type': self.order_type,
                'trader_link': self.trader}


class ArbitrageList:
    """
    Queue-like container for arbitrage orders submitted during one simulation iteration.
    """

    def __init__(self):
        self.arbitrage_list: list[ArbitrageOrder] = []

    def add(self, arbitrage_order: ArbitrageOrder):
        """
        Add arbitrage order to queue.
        """
        self.arbitrage_list.append(arbitrage_order)

    def simulate(self):
        """
        Execute queued arbitrage orders by priority and clear queue at the end.
        """
        self.arbitrage_list.sort()
        for order in self.arbitrage_list:
            cur_q, cur_min_cash, cur_pay_prio, cur_order_type, cur_trader = order.qty, order.min_got_cash, order.pay_for_prio, order.order_type, order.trader

            # by default buy => buy on cex, sell on dex
            #            sell => sell on cex, by on dex
            eps = 10 ** (-5)
            if cur_order_type == "buy":
                compl_q, spent_cash, got_cash = cur_trader.get_quotes_buy(cur_q)
                cur_profit = got_cash - spent_cash - cur_trader.amm.gas_cost - cur_pay_prio
                if compl_q == cur_q and got_cash >= cur_min_cash and cur_profit > 0 + eps:
                    cur_trader.buy_market(cur_q)
                    cur_trader.amm.sell(cur_q)
                    cur_trader.cash += got_cash
                    cur_trader.assets -= cur_q
                    cur_trader.cash -= cur_trader.amm.gas_cost + cur_pay_prio
            elif cur_order_type == "sell":
                compl_q, spent_cash, got_cash = cur_trader.get_quotes_sell(cur_q)
                cur_profit = got_cash - spent_cash - cur_trader.amm.gas_cost - cur_pay_prio
                if compl_q == cur_q and got_cash >= cur_min_cash and cur_profit > 0 + eps:
                    cur_trader.amm.buy(cur_q)
                    cur_trader.cash -= spent_cash
                    cur_trader.assets += cur_q
                    cur_trader.sell_market(cur_q)
                    cur_trader.cash -= cur_trader.amm.gas_cost + cur_pay_prio
        self.arbitrage_list.clear()


class AMMAgent(ABC):
    """
    Base class for AMM implementations used in simulation.
    """
    id = 0

    def __init__(self, price: float or int = 100,
                 gas_cost: float = 0, fee: float = 0.003, asset: float or int = 1000):
        """
        :param price: initial AMM spot price
        :param gas_cost: fixed transaction cost for one AMM trade
        :param fee: AMM fee rate charged on swap
        :param asset: initial amount of asset in AMM pool
        """
        self.name = f'AMMAgent{self.id}'
        AMMAgent.id += 1
        self.gas_cost = gas_cost
        self.fee = fee
        self.cash = asset * price
        self.asset = asset
        self.arbitrage_queue = ArbitrageList()

    @abstractmethod
    def try_buy(self, amount_in):
        """
        Return required cash without mutating AMM state.
        """
        ...

    @abstractmethod
    def buy(self, amount_in):
        """
        Execute buy operation and mutate AMM reserves.
        """
        ...

    @abstractmethod
    def try_sell(self, amount_in):
        """
        Return expected cash output without mutating AMM state.
        """
        ...

    @abstractmethod
    def sell(self, amount_in):
        """
        Execute sell operation and mutate AMM reserves.
        """
        ...

    @abstractmethod
    def price(self):
        """
        :return: current AMM spot price
        """
        ...


class UniswapV2(AMMAgent):
    """
    Constant product AMM with fee, compatible with Uniswap V2 pricing formulas.
    """

    def price(self):
        """
        :return: spot price of one asset unit in cash units
        """
        return self.cash / self.asset

    def try_buy(self, amount_out):
        """
        Preview cash required to buy amount_out from AMM.
        """
        cash_in = (self.cash * amount_out) / ((self.asset - amount_out) * (1 - self.fee))
        return cash_in

    def buy(self, amount_out):
        """
        Buy amount_out from AMM and update reserves.
        """
        cash_in = self.try_buy(amount_out)
        self.cash += cash_in
        self.asset -= amount_out
        return cash_in

    def try_sell(self, amount_in):
        """
        Preview cash output for selling amount_in to AMM.
        """
        no_fee = amount_in * (1 - self.fee)
        cash_out = (self.cash * no_fee) / (self.asset + no_fee)
        return cash_out

    def sell(self, amount_in):
        """
        Sell amount_in to AMM and update reserves.
        """
        cash_out = self.try_sell(amount_in)
        self.asset += amount_in
        self.cash -= cash_out
        return cash_out

    def token_supply(self, asset_in):
        """
        Add proportional liquidity while preserving current spot price.
        """
        self.cash += (self.cash / self.asset) * asset_in
        self.asset += asset_in
