from AgentBasedModel import *
from AgentBasedModel.amms.amms import UniswapV2
from AgentBasedModel.utils.math import *

import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANGE = list(range(0, 6))
WINDOW = 5
SIZE = 10
FUNCS = [
    ('price', lambda info, w: info.prices)
]

traders = list()
before = list()
after = list()

for n_rand, n_fund, n_chart, n_univ in tqdm(list(itertools.product(RANGE, repeat=4))):
    for is_mm in range(2):
        exchange = ExchangeAgent(volume=1000)
        amm = UniswapV2()
        simulator = Simulator(**{
            'exchange': exchange,
            'amm': amm,
            'traders': [
                *[Random(exchange, 10 ** 3) for _ in range(n_rand)],
                *[Fundamentalist(exchange, 10 ** 3) for _ in range(n_fund)],
                *[Chartist(exchange, 10 ** 3) for _ in range(n_chart)],
                *[Universalist(exchange, 10 ** 3) for _ in range(n_univ)],
                *[MarketMaker(exchange, 10 ** 3) for _ in range(is_mm)]
            ],
            'arbitrage_traders': [*[Arbitrage(exchange, 10**3, amm) for _ in range(5)]],
            'events': [MarketPriceShock(200, -10)]
        })
        info = simulator.info
        simulator.simulate(500, silent=True)

        shock_key = str(simulator.events[0])
        tmp = aggToShock(simulator, 1, FUNCS)[shock_key]['price']

        traders.append({'Random': n_rand, 'Fundamentalist': n_fund, 'Chartist': n_chart, 'Universalist': n_univ,
                        'MarketMaker': is_mm})
        before.append(tmp['right before'])
        after.append(tmp['after'])
        plot_price(simulator.info, spread=False)
        plot_dividend(simulator.info)
        plot_volatility_price(simulator.info)