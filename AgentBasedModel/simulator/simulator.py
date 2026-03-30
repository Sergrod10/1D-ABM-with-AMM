from AgentBasedModel import Arbitrage
from AgentBasedModel.amms import AMMAgent
from AgentBasedModel.agents import ExchangeAgent, Universalist, Chartist, Fundamentalist
from AgentBasedModel.utils.math import mean, std, difference, rolling
import random
from tqdm import tqdm
import numpy as np


class Simulator:
    """
    Simulator is responsible for launching agents' actions and executing scenarios
    """
    def __init__(self, exchange: ExchangeAgent = None, amm: AMMAgent = None, traders: list = None, arbitrage_traders: list = None, events: list = None, avg_traders=0, last_step=0, last_ret=0, noisy_level=0, norm_coef_lr=1):
        """
        :param exchange: central limit order book market agent
        :param amm: AMM venue used for arbitrage, optional
        :param traders: list of standard market traders
        :param arbitrage_traders: list of arbitrage traders interacting with AMM
        :param events: list of scenario events
        :param traders_ratio: fraction of standard traders that trade on each tick (0..1)
        """
        self.exchange = exchange
        self.amm = amm
        self.events = [event.link(self) for event in events] if events else None  # link all events to simulator
        self.traders = traders
        self.arbitrage_traders = arbitrage_traders
        self.info = SimulatorInfo(self.exchange, self.traders)  # links to existing objects
        self.ratios = []
        self.avg_traders = avg_traders
        self.last_step = last_step
        self.last_ret = last_ret
        self.noisy_level = noisy_level
        self.norm_coef_lr = norm_coef_lr
        if amm is not None:
            self.amm_info = AmmAgentInfo(self.amm, self.arbitrage_traders)

    def _payments(self):
        for trader in self.traders:
            # Dividend payments
            trader.cash += trader.assets * self.exchange.dividend()  # allow negative dividends
            # Interest payment
            trader.cash += trader.cash * self.exchange.risk_free  # allow risk-free loan

        if self.amm is not None:
            for trader in self.arbitrage_traders:
                # Dividend payments
                trader.cash += trader.assets * self.exchange.dividend()  # allow negative dividends
                # Interest payment
                trader.cash += trader.cash * self.exchange.risk_free  # allow risk-free loan

    # генерация сколько агентов будет участвовать, зависит от среднего, ластового значения и ластовой доходности и стандартного шума
    # ф-ией для получения процентов используется сигмоида, так как она как раз от 0 до 1, а также супер удобная для подбора параметров
    def get_cur_traders_count(self):
        coef1 = self.avg_traders
        coef2 = 0
        if len(self.ratios) > 0:
            coef2 = self.last_step * (self.ratios[-1] - self.avg_traders)
        coef3 = 0
        if len(self.info.prices) > 1:
            coef3 = self.last_ret * (abs(self.info.prices[-1] - self.info.prices[-2]) / self.info.prices[-2]) * self.norm_coef_lr
        coef4 = self.noisy_level * np.random.normal(0.0, 1.0)
        coef = coef1 + coef2 + coef3 + coef4
        self.ratios.append(coef)
        return 1 / (1 + np.exp(-coef))

    def simulate(self, n_iter: int, silent=False) -> object:
        for it in tqdm(range(n_iter), desc='Simulation', disable=silent):
            # Call scenario
            if self.events:
                for event in self.events:
                    event.call(it)

            # Capture current info
            self.info.capture()
            if self.amm is not None:
                self.amm_info.capture()

            # Change behaviour
            for trader in self.traders:
                if type(trader) == Universalist:
                    trader.change_strategy(self.info)
                elif type(trader) == Chartist:
                    trader.change_sentiment(self.info)

            # Call Traders
            random.shuffle(self.traders)
            ratio = self.get_cur_traders_count()
            for trader in self.traders[:max(1, round(ratio * len(self.traders)))]:
                trader.call()

            if self.amm is not None:
                random.shuffle(self.arbitrage_traders)
                for trader in self.arbitrage_traders:
                    trader.call()
                self.amm_info.capture_queue()
                self.amm.arbitrage_queue.simulate()

            # Payments and dividends
            self._payments()  # pay dividends
            self.exchange.generate_dividend()  # generate next dividends

        return self


class SimulatorInfo:
    """
    SimulatorInfo is responsible for capturing data during simulating
    """

    def __init__(self, exchange: ExchangeAgent = None, traders: list = None):
        self.exchange = exchange
        self.traders = {t.id: t for t in traders}

        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.dividends = list()  # dividend paid at each iteration
        self.orders = list()  # order book statistics

        # Agent statistics
        self.equities = list()  # agent: equity
        self.cash = list()  # agent: cash
        self.assets = list()  # agent: number of assets
        self.types = list()  # agent: current type
        self.sentiments = list()  # agent: current sentiment
        self.returns = [{tr_id: 0 for tr_id in self.traders.keys()}]  # agent: iteration return

        """
        # Market Statistics
        self.prices = list()  # price at the end of iteration
        self.spreads = list()  # bid-ask spreads
        self.spread_sizes = list()  # bid-ask spread sizes
        self.dividends = list()
        self.orders_quantities = list()  # list -> (bid, ask)
        self.orders_volumes = list()  # list -> (bid, ask) -> (sum, mean, q1, q3, std)
        self.orders_prices = list()  # list -> (bid, ask) -> (mean, q1, q3, std)

        # Agent Statistics
        self.equity = list()  # sum of equity of agents
        self.cash = list()  # sum of cash of agents
        self.assets_qty = list()  # sum of number of assets of agents
        self.assets_value = list()  # sum of value of assets of agents
        """

    def capture(self):
        """
        Method called at the end of each iteration to capture basic info on simulation.

        **Attributes:**

        *Market Statistics*

        - :class:`list[float]` **prices** --> stock prices on each iteration
        - :class:`list[dict]` **spreads** --> order book spreads on each iteration
        - :class:`list[float]` **dividends** --> dividend paid on each iteration
        - :class:`list[dict[dict]]` **orders** --> order book price, volume, quantity stats on each iteration

        *Traders Statistics*

        - :class:`list[dict]` **equities** --> each agent's equity on each iteration
        - :class:`list[dict]` **cash** --> each agent's cash on each iteration
        - :class:`list[dict]` **assets** --> each agent's number of stocks on each iteration
        - :class:`list[dict]` **types** --> each agent's type on each iteration
        """
        # Market Statistics
        spread = self.exchange.spread()
        if spread is None:
            bid = self.exchange.order_book['bid'].first
            ask = self.exchange.order_book['ask'].first
            if bid is not None:
                fallback_price = bid.price
            elif ask is not None:
                fallback_price = ask.price
            elif self.prices:
                fallback_price = self.prices[-1]
            else:
                fallback_price = 100.0
            spread = {'bid': fallback_price, 'ask': fallback_price}

        self.prices.append((spread['bid'] + spread['ask']) / 2)
        self.spreads.append(spread)
        self.dividends.append(self.exchange.dividend())
        self.orders.append({
            'quantity': {'bid': len(self.exchange.order_book['bid']), 'ask': len(self.exchange.order_book['ask'])},
            # 'price mean': {
            #     'bid': mean([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.price for order in self.exchange.order_book['ask']])},
            # 'price std': {
            #     'bid': std([order.price for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.price for order in self.exchange.order_book['ask']])},
            # 'volume sum': {
            #     'bid': sum([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': sum([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume mean': {
            #     'bid': mean([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': mean([order.qty for order in self.exchange.order_book['ask']])},
            # 'volume std': {
            #     'bid': std([order.qty for order in self.exchange.order_book['bid']]),
            #     'ask': std([order.qty for order in self.exchange.order_book['ask']])}
        })

        # Trader Statistics
        self.equities.append({t_id: t.equity() for t_id, t in self.traders.items()})
        self.cash.append({t_id: t.cash for t_id, t in self.traders.items()})
        self.assets.append({t_id: t.assets for t_id, t in self.traders.items()})
        self.types.append({t_id: t.type for t_id, t in self.traders.items()})
        self.sentiments.append({t_id: t.sentiment for t_id, t in self.traders.items() if t.type == 'Chartist'})
        self.returns.append({tr_id: (self.equities[-1][tr_id] - self.equities[-2][tr_id]) / self.equities[-2][tr_id]
                             for tr_id in self.traders.keys()}) if len(self.equities) > 1 else None

    def fundamental_value(self, access: int = 1) -> list:
        divs = self.dividends.copy()
        n = len(divs)  # number of iterations
        divs.extend(self.exchange.dividend(access)[1:access])  # add not recorded future divs
        r = self.exchange.risk_free

        return [Fundamentalist.evaluate(divs[i:i+access], r) for i in range(n)]

    def stock_returns(self, roll: int = None) -> list or float:
        p = self.prices
        div = self.dividends
        r = [(p[i+1] - p[i]) / p[i] + div[i] / p[i] for i in range(len(p) - 1)]
        return rolling(r, roll) if roll else mean(r)

    def abnormal_returns(self, roll: int = None) -> list:
        rf = self.exchange.risk_free
        r = [r - rf for r in self.stock_returns()]
        return rolling(r, roll) if roll else r

    def return_volatility(self, window: int = None) -> list or float:
        if window is None:
            return std(self.stock_returns())
        n = len(self.stock_returns(1))
        return [std(self.stock_returns(1)[i:i+window]) for i in range(n - window)]

    def price_volatility(self, window: int = None) -> list or float:
        if window is None:
            return std(self.prices)
        return [std(self.prices[i:i+window]) for i in range(len(self.prices) - window)]

    def liquidity(self, roll: int = None) -> list or float:
        n = len(self.prices)
        spreads = [el['ask'] - el['bid'] for el in self.spreads]
        prices = self.prices
        liq = [spreads[i] / prices[i] for i in range(n)]
        return rolling(liq, roll) if roll else mean(liq)


class AmmAgentInfo:
    """
    AmmAgentInfo captures AMM state, arbitrage queue state, and arbitrage trader state on each iteration.
    """
    def __init__(self, amm: AMMAgent = None, traders: list[Arbitrage] = None):
        """
        :param amm: AMM agent linked to simulator
        :param traders: arbitrage traders linked to simulator
        """
        self.amm = amm
        self.traders = {t.id: t for t in traders}

        # AMM Statistics
        self.equities_amm = list()
        self.prices = list()
        self.cash_amm = list()
        self.assets_amm = list()
        self.arbs_queue = list()

        # Agent statistics
        self.equities = list()  # agent: equity
        self.cash_traders = list()  # agent: cash
        self.assets_traders = list()  # agent: number of assets
        self.participated = list()

    def capture(self):
        """
        Capture AMM reserves/price and current arbitrage traders metrics.
        """
        # AMM Statistics
        self.prices.append(self.amm.price())
        self.cash_amm.append(self.amm.cash)
        self.assets_amm.append(self.amm.asset)
        self.equities_amm.append(self.amm.price() * self.amm.asset + self.amm.cash)
        self.capture_queue()

        # Trader Statistics
        self.equities.append({t_id: t.equity() for t_id, t in self.traders.items()})
        self.cash_traders.append({t_id: t.cash for t_id, t in self.traders.items()})
        self.assets_traders.append({t_id: t.assets for t_id, t in self.traders.items()})
        self.participated.append({t_id: (t.buy, t.sell) for t_id, t in self.traders.items()})

    def capture_queue(self):
        """
        Store state of arbitrage queue on current iteration
        """
        current_orders = []
        for order in self.amm.arbitrage_queue.arbitrage_list:
            current_orders.append({
                'order_id': order.order_id,
                'order_type': order.order_type,
                'qty': order.qty,
                'min_got_cash': order.min_got_cash,
                'pay_for_prio': order.pay_for_prio,
                'trader_id': order.trader.id if order.trader is not None else None,
                'trader_name': order.trader.name if order.trader is not None else None,
            })

        self.arbs_queue.append({
            'quantity': len(current_orders),
            'orders': current_orders,
        })
