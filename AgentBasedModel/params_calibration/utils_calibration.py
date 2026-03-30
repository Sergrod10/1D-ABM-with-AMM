import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
import matplotlib.pyplot as plt

def mean_return(returns):
    return np.mean(returns)

def std_return(returns):
    return np.std(returns)

def skewness_return(returns):
    return skew(returns)

def kurtosis_return(returns):
    return kurtosis(returns)

def min_return(returns):
    return np.min(returns)

def max_return(returns):
    return np.max(returns)

def autocorrelation_at_lag(returns, lag):
    if lag == 0:
        return 1.0
    returns_mean = mean_return(returns)
    num = np.sum((returns[lag:] - returns_mean) * (returns[:-lag] - returns_mean))
    denom = np.sum((returns - returns_mean) ** 2)
    return num / denom

def autocorr_lags(returns, lags):
    return np.array([autocorrelation_at_lag(returns, lag) for lag in lags])

def quan(returns, q):
    return np.quantile(returns, q)

# returns - доходности с тиком 10 секунд, returns2 - с большим тиком, например, 5 минут, чтобы не шумели сильно макс/мин std, среднее и куртозис
def get_all_params(returns, returns2):
    mean_r = mean_return(returns2)
    std_r = std_return(returns2)
    min_r = min_return(returns2)
    max_r = max_return(returns2)
    skew_r = skewness_return(returns)
    kurt_r = kurtosis_return(returns2)
    lags = np.arange(0, 21)
    corr_ret = autocorr_lags(returns, lags)
    corr_abs_ret = autocorr_lags(np.abs(returns), lags)
    return {"mean_on_ret2" : mean_r, "std_on_ret2" : std_r, "min_on_ret2" : min_r, "max_on_ret2" : max_r, "skewness_on_ret" : skew_r, "kurtosis_on_ret2" : kurt_r, "autocorrelation_on_ret" : corr_ret, "autocorrelation_on_abs_ret" : corr_abs_ret}

def calc_returns(prices):
    return prices[1:] / prices[:-1] - 1

def get_prices(path) -> np.ndarray:
    df = pd.read_csv(path, usecols=["close"])
    arr = pd.to_numeric(df["close"])
    return arr.to_numpy()

# красивый принт
def print_params(params) -> None:
    print("ETHUSDT stylized facts")
    print("=" * 72)

    return_metrics = {"mean_on_ret2", "std_on_ret2", "min_on_ret2", "max_on_ret2"}
    core_keys = [
        "mean_on_ret2",
        "std_on_ret2",
        "min_on_ret2",
        "max_on_ret2",
        "skewness_on_ret",
        "kurtosis_on_ret2",
    ]
    core_rows = []
    for key in core_keys:
        value = float(params[key])
        if key in return_metrics:
            value_view = f"{value:.6f} ({value * 100:.3f}%)"
        else:
            value_view = f"{value:.6f}"
        core_rows.append({"metric": key, "value": value_view})

    core_df = pd.DataFrame(core_rows)
    print(core_df.to_string(index=False))

    lags = np.arange(len(params["autocorrelation_on_ret"]))
    acorr_df = pd.DataFrame({
        "lag": lags,
        "autocorr_ret": params["autocorrelation_on_ret"],
        "autocorr_abs_ret": params["autocorrelation_on_abs_ret"],
    })
    print("\nAutocorrelation by lag")
    print("-" * 72)
    print(acorr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 72)

def plot_autocorr(params: dict) -> None:
    lags = np.arange(len(params["autocorrelation_on_ret"]))

    plt.figure(figsize=(10, 5))
    plt.scatter(lags, params["autocorrelation_on_ret"], s=28)
    plt.title("ETHUSDT: autocorrelation of returns by lag")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(lags, params["autocorrelation_on_abs_ret"], s=28)
    plt.title("ETHUSDT: autocorrelation of abs(returns) by lag")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(alpha=0.3)
    plt.show()

def pipeline(prices_10s, is_print = 1):
    returns_10s = calc_returns(prices_10s)
    returns_5m = calc_returns(get_prices_5m(prices_10s))
    params = get_all_params(returns_10s, returns_5m)
    if is_print:
        print_params(params)
    return params

def get_prices_5m(prices_10s):
    prices_10s = np.asarray(prices_10s)
    return prices_10s[29::30]

