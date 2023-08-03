# --------------------------------  NOTE  ----------------------------------------
# 1 This code is to quantify uncertainty based on multiple time series;
# 2 The format of the input data is [ts ts1 ...tsN];
# 3 Coder: Honglin Li        Date: 08/03/2023       @ UT-Dallas
# --------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import antropy as ant  # Make sure to install the `antropy` library if you haven't already
import CRPS.CRPS as pscore
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]


class UncertaintyQualificationMetrics(object):
    def __init__(self, data, window_size=12):
        """
        Initialize the UncertaintyQualificationMetrics class with data, window size, and percentiles.

        Parameters:
            data (pandas.DataFrame): DataFrame containing true and predicted values for each window.
            window_size (int): Size of the window used for NRMSE and NMAE calculations.
        """
        self.data = data
        self.window_size = window_size

    def data_preprocess(self):
        """
        Preprocess the input data.

        Returns:
            pandas.DataFrame: DataFrame containing true and predicted values for each window.
        """
        data = self.data
        data = data.dropna()
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def calculate_percentile(samples, percentile):
        sorted_samples = sorted(samples)
        n = len(samples)
        index = (percentile / 100) * (n + 1)
        if index.is_integer():
            percentile_value = sorted_samples[int(index) - 1]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            lower_value = sorted_samples[lower_index - 1]
            upper_value = sorted_samples[upper_index - 1]
            percentile_value = (lower_value + upper_value) / 2
        return percentile_value

    def variogram(h, nugget, range_, sill):
        return nugget + sill * (1 - np.exp(-3 * h ** 2 / range_ ** 2))

    def fit_variogram(h, gamma, func=variogram):
        popt, _ = curve_fit(func, h, gamma)
        return popt


class SingleTimeSeriesMovingWindowMetrics(UncertaintyQualificationMetrics):
    def __init__(self, data, window_size):

        """
        Initialize the Single Time Series Moving-Window class.

        Args:
            data (numpy.array or list): The input data.
            window_size (int): The size of the window for analysis.
        """
        super(SingleTimeSeriesMovingWindowMetrics, self).__init__(data, window_size)

    def calculate_entropy(self, window):
        return ant.spectral_entropy(window, sf=100, method='welch', normalize=True)

    def entropy(self):
        """
        Calculate Spectral Entropy for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Spectral Entropy values for each window.
        """
        entropy_df = pd.DataFrame(columns=['Entropy'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = self.calculate_entropy(window)
            entropy_df.loc[i] = result
        return entropy_df

    def standard_deviation(self):
        """
        Calculate Standard Deviation for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Standard Deviation values for each window.
        """
        std_df = pd.DataFrame(columns=['Standard Deviation'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = np.std(window)
            std_df.loc[i] = result
        return std_df

    def turbulence_intensity(self):
        """
        Calculate Turbulence Intensity for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Turbulence Intensity values for each window.
        """
        tbl_df = pd.DataFrame(columns=['Turbulence Intensity'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = np.std(window) / (np.mean(window) + 0.001)
            tbl_df.loc[i] = result
        return tbl_df

    def variability_index(self):
        """
        Calculate Variability Index for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Variability Index values for each window.
        """
        var_df = pd.DataFrame(columns=['Variability Index'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = (np.max(window) - np.min(window)) / (np.mean(window) + 0.001)
            var_df.loc[i] = result
        return var_df


class EnsembleMember(UncertaintyQualificationMetrics):
    def __init__(self, data, window_size=12):
        """
        Initialize the Single Time Series Moving-Window class.

        Args:
            data (2-D DataFrame): The input data. # TODO: ensemble members more than 2
            window_size (int): The size of the window for analysis.
        """
        super(EnsembleMember, self).__init__(data, window_size)

    def correlation(self):
        """
        Calculate Correlation for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Correlation values for each window.
        """
        corr_df = pd.DataFrame(columns=['Correlation'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = window.corr().iloc[0, 1]
            corr_df.loc[i] = result
        return corr_df

    def mape(self):
        """
        Calculate Mean Absolute Percentage Error for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Mean Absolute Percentage Error values for each window.
        """
        mape_df = pd.DataFrame(columns=['MAPE'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            mask = window.iloc[:, 0] != 0
            mape_df.loc[i] = (np.fabs(window.iloc[:, 0] - window.iloc[:, 1]) / window.iloc[:, 0])[mask].mean()
        return mape_df

    def nrmse(self):
        """
        Calculate Normalized Root Mean Squared Error for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Normalized Root Mean Squared Error values for each window.
        """
        nrmse_df = pd.DataFrame(columns=['NRMSE'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = np.sqrt(((window.iloc[:, 0] - window.iloc[:, 1]) ** 2).mean()) / \
                     (np.max(window.iloc[:, 0]) - np.min(window.iloc[:, 0]))
            nrmse_df.loc[i] = result
        return nrmse_df

    def nmae(self):
        """
        Calculate Normalized Mean Absolute Error for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Normalized Mean Absolute Error values for each window.
        """
        nmae_df = pd.DataFrame(columns=['NMAE'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = np.mean(np.abs(window.iloc[:, 0] - window.iloc[:, 1])) / \
                     (np.max(window.iloc[:, 0]) - np.min(window.iloc[:, 0]))
            nmae_df.loc[i] = result
        return nmae_df

    def spread_index(self):
        """
        Calculate Spread Index for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Spread Index values for each window.
        """

        percentile_df = pd.DataFrame(columns=[f"{p}th percentile" for p in percentiles])

        for index, row in self.data.iterrows():
            samples = row.tolist()
            row_percentiles = [self.calculate_percentile(samples, p) for p in percentiles]
            percentile_df.loc[index] = row_percentiles

        differences = []
        for i in range(1, 5):
            diff = percentile_df.iloc[:, -i] - percentile_df.iloc[:, i - 1]
            differences.append(diff)

        row_mean = self.data.mean(axis=1)

        return pd.DataFrame({
            f'SI_{100 - i * 10}_{i * 10}': diff / row_mean
            for i, diff in enumerate(differences, start=1)
        })

    def predictablity_index(self):
        """
        Calculate Predictablity Index for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Predictablity Index values for each window.
        """
        SI = self.spread_index()
        PI = pd.DataFrame(columns=['PI_90_10', 'PI_80_20', 'PI_70_30', 'PI_60_40'])

        for i in range(len(SI) - self.window_size):
            window = SI.iloc[i:i + self.window_size, :]
            results = window.mean() - window.iloc[0, :]
            return PI.append(results, ignore_index=True)

    def crps(self):
        """
        Calculate Continuous Ranked Probability Score for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Continuous Ranked Probability Score values for each window.
        """

        # # Calculate quantiles for each row
        # ens_quantile = self.data.apply(lambda row: np.quantile(row, q=np.arange(0.01, 1.0, 0.01)), axis=1)
        #
        # # Rename columns to 'Q1', 'Q2', ..., 'Q99'
        # ens_quantile.columns = ['Q' + str(i) for i in range(1, 100)]
        #
        # ens_quantile = pd.DataFrame(ens_quantile.tolist())
        #
        # # Calculate standard deviation
        # stand_dev = (ens_quantile['Q99'] - ens_quantile['Q50']) / np.quantile(np.random.normal(size=100000), 0.99)
        # stand_dev[stand_dev <= 0] = 1e-10

        return pscore(self.data, self.data['Actual']).compute()

        # Calculate nCRPS
        # nCRPS = np.mean(CRPS_all) / max(df['Actual'])

    def veritication_rank_histogram(self):
        """
        Calculate Veritication Rank Histogram for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Veritication Rank Histogram values for each window.
        """
        pass


class MultipleTimeSeries(UncertaintyQualificationMetrics):
    def __init__(self, data, window_size=12):
        """
        Initialize the Multiple Time Series Moving-Window class.

        Args:
            data (numpy.array or list): The input data.
            window_size (int): The size of the window for analysis.
        """
        super(MultipleTimeSeries, self).__init__(data, window_size)

    def correlation(self):
        """
        Calculate Correlation for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Correlation values for each window.
        """
        corr_df = pd.DataFrame(columns=['Correlation'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            result = window.corr().iloc[0, 1]
            corr_df.loc[i] = result
        return corr_df

    def mape(self):
        """
        Calculate Mean Absolute Percentage Error for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Mean Absolute Percentage Error values for each window.
        """
        mape_df = pd.DataFrame(columns=['MAPE'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size]
            mask = window.iloc[:, 0] != 0
            mape_df.loc[i] = (np.fabs(window.iloc[:, 0] - window.iloc[:, 1]) / window.iloc[:, 0])[mask].mean()
        return mape_df

    def spread_index(self):
        """
        Calculate Spread Index for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Spread Index values for each window.
        """
        percentile_df = pd.DataFrame(columns=[f"{p}th percentile" for p in percentiles])

        for index, row in self.data.iterrows():
            samples = row.tolist()
            row_percentiles = [self.calculate_percentile(samples, p) for p in percentiles]
            percentile_df.loc[index] = row_percentiles

        differences = []
        for i in range(1, 5):
            diff = percentile_df.iloc[:, -i] - percentile_df.iloc[:, i - 1]
            differences.append(diff)

        row_mean = self.data.mean(axis=1)

        return pd.DataFrame({
            f'SI_{100 - i * 10}_{i * 10}': diff / row_mean
            for i, diff in enumerate(differences, start=1)
        })

    def predictablity_index(self):
        """
        Calculate Predictablity Index.

        Returns:
            pandas.DataFrame: DataFrame containing Predictablity Index values for each window.
        """
        SI = self.spread_index()
        PI = pd.DataFrame(columns=['PI_90_10', 'PI_80_20', 'PI_70_30', 'PI_60_40'])

        for i in range(len(SI) - self.window_size):
            window = SI.iloc[i:i + self.window_size, :]
            results = window.mean() - window.iloc[0, :]
            return PI.append(results, ignore_index=True)

    def kriging_variogram(self, windspeed_cols, latitude_col, longitude_col):
        """
        Calculate Kriging Variogram.

        Returns:
            pandas.DataFrame: DataFrame containing Kriging Variogram values for each window.
        """

        br = np.arange(0, 0.201, 0.005)  # farm lat/lon degree; this is for Cedar Creek wind farm
        ini_vals = np.array(np.meshgrid(br, br)).T.reshape(-1, 2)

        nuggets = []
        ranges = []
        sills = []

        for i in range(len(self.data)):
            df_cc = self.data[[windspeed_cols[i], latitude_col, longitude_col]]
            geo_cc = df_cc[[longitude_col, latitude_col]].to_numpy()
            data_cc = df_cc[windspeed_cols[i]].to_numpy()
            dist_matrix = squareform(pdist(geo_cc, metric='euclidean'))

            variogram_vals = 0.5 * (dist_matrix ** 2).flatten()
            data_diff_matrix = np.abs(data_cc[:, None] - data_cc[None, :]).flatten()
            variogram_vals += data_diff_matrix ** 2

            popt = self.fit_variogram(dist_matrix.flatten(), variogram_vals)

            nuggets.append(popt[0])
            ranges.append(popt[1])
            sills.append(popt[2])

        result = {'nugget': nuggets, 'range': ranges, 'sill': sills}

        return result
