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

    @staticmethod
    def variogram(h, nugget, range_, sill):
        return nugget + sill * (1 - np.exp(-3 * h ** 2 / range_ ** 2))

    @staticmethod
    def fit_variogram(h, gamma, func=variogram):
        popt, _ = curve_fit(func, h, gamma)
        return popt


class SingleTimeSeriesMovingWindowMetrics(UncertaintyQualificationMetrics):
    def __init__(self, data, window_size):

        """
        Initialize the Single Time Series Moving-Window class.
        The input data for the demo is Argonne_era5_2018.csv

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
            list: List containing Spectral Entropy values for each window.
        """
        entropy_df = []
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1].values.flatten()
            entropy_df.append(self.calculate_entropy(window))
        return entropy_df

    def standard_deviation(self):
        """
        Calculate Standard Deviation for each window.

        Returns:
            list: List containing Standard Deviation values for each window.
        """
        std_df = []
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1].values
            std_df.append(np.std(window))
        return std_df

    def turbulence_intensity(self):
        """
        Calculate Turbulence Intensity for each window.

        Returns:
            list: List containing Turbulence Intensity values for each window.
        """
        tbl_df = []
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1].values
            tbl_df.append(np.std(window) / np.mean(window))
        return tbl_df

    def variability_index(self):
        """
        Calculate Variability Index for each window.

        Returns:
            list: List containing Variability Index values for each window.
        """
        Variability_Index = []
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1].values
            result = (np.max(window) - np.min(window)) / np.mean(window)
            Variability_Index.append(result)
        return Variability_Index


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
        The input data for the demo is plant_level_hourly_wind_power_forecast.csv

        Returns:
            pandas.DataFrame: DataFrame containing Correlation values for each window.
        """
        corr_df = pd.DataFrame(columns=['Correlation'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1]
            result = window.corr().iloc[0, 1]
            corr_df.loc[i] = result
        return corr_df

    def mape(self):
        """
        Calculate Mean Absolute Percentage Error for each window.
        The input data for the demo is plant_level_hourly_wind_power_forecast.csv

        Returns:
            pandas.DataFrame: DataFrame containing Mean Absolute Percentage Error values for each window.
        """
        mape_df = pd.DataFrame(columns=['MAPE'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1]
            mask = window.iloc[:, 0] != 0
            mape_df.loc[i] = (np.fabs(window.iloc[:, 0] - window.iloc[:, 1]) / window.iloc[:, 0])[mask].mean()
        return mape_df

    def nrmse(self):
        """
        Calculate Normalized Root Mean Squared Error for each window.
        The input data for the demo is plant_level_hourly_wind_power_forecast.csv

        Returns:
            pandas.DataFrame: DataFrame containing Normalized Root Mean Squared Error values for each window.
        """
        nrmse_values = []
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1]
            rmse = np.sqrt(((window.iloc[:, 0] - window.iloc[:, 1]) ** 2).mean())
            data_range = np.max(window.iloc[:, 0]) - np.min(window.iloc[:, 0])
            if data_range != 0:
                nrmse_values.append(rmse / data_range)
            else:
                nrmse_values.append(rmse / 0.001)
        nrmse_df = pd.DataFrame({'NRMSE': nrmse_values})
        return nrmse_df

    def nmae(self):
        """
        Calculate Normalized Mean Absolute Error for each window.
        The input data for the demo is plant_level_hourly_wind_power_forecast.csv

        Returns:
            pandas.DataFrame: DataFrame containing Normalized Mean Absolute Error values for each window.
        """
        nmae_df = []  # pd.DataFrame(columns=['NMAE'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1]
            mae = np.mean(np.abs(window.iloc[:, 0] - window.iloc[:, 1]))
            data_range = (np.max(window.iloc[:, 0]) - np.min(window.iloc[:, 0]))
            if data_range != 0:
                nmae_df.append(mae / data_range)
            else:
                nmae_df.append(mae / 0.001)

        nmae_df = pd.DataFrame({'NMAE': nmae_df})
        return nmae_df

    def spread_index(self):
        """
        Calculate Spread Index.
        Input data should be a DataFrame with more than two columns, each column representing one ensemble member.
        The input data for the demo is PaloDuro.csv

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
        The input data for the demo is PaloDuro.csv

        Returns:
            pandas.DataFrame: DataFrame containing Predictablity Index values for each window.
        """
        SI = self.spread_index()
        pi = []
        for i in range(len(SI) - self.window_size):
            window = SI.iloc[i:i + self.window_size, :]
            results = (window.mean() - window.iloc[0, :]).values
            pi.append(results)
        return pd.DataFrame(pi, columns=['PI_90_10', 'PI_80_20', 'PI_70_30', 'PI_60_40'])

    def crps(self):
        """
        Calculate Continuous Ranked Probability Score for each window.
        The input data for the demo is PaloDuro.csv

        Returns:
            pandas.DataFrame: DataFrame containing Continuous Ranked Probability Score values for each window.
        """
        crps = []
        for i, row in self.data.iterrows():
            res = pscore(row[:9], row[-1]).compute()
            crps.append(res[0])
        crps_df = pd.DataFrame({'CRPS': crps})
        return crps_df


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
        The input data for the demo is plant_level_hourly_wind_power_forecast.csv

        Returns:
            pandas.DataFrame: DataFrame containing Correlation values for each window.
        """
        corr_df = pd.DataFrame(columns=['Correlation'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1]
            result = window.corr().iloc[0, 1]
            corr_df.loc[i] = result
        return corr_df

    def mape(self):
        """
        Calculate Mean Absolute Percentage Error for each window.
        The input data for the demo is plant_level_hourly_wind_power_forecast.csv

        Returns:
            pandas.DataFrame: DataFrame containing Mean Absolute Percentage Error values for each window.
        """
        mape_df = pd.DataFrame(columns=['MAPE'])
        for i in range(len(self.data) - self.window_size + 1):
            window = self.data[i:i + self.window_size - 1]
            mask = window.iloc[:, 0] != 0
            mape_df.loc[i] = (np.fabs(window.iloc[:, 0] - window.iloc[:, 1]) / window.iloc[:, 0])[mask].mean()
        return mape_df

    def spread_index(self):
        """
        Calculate Spread Index for each window.
        The input data for the demo is PaloDuro.csv

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
        The input data for the demo is PaloDuro.csv

        Returns:
            pandas.DataFrame: DataFrame containing Predictablity Index values for each window.
        """
        SI = self.spread_index()
        pi = []
        for i in range(len(SI) - self.window_size):
            window = SI.iloc[i:i + self.window_size, :]
            results = (window.mean() - window.iloc[0, :]).values
            pi.append(results)
        return pd.DataFrame(pi, columns=['PI_90_10', 'PI_80_20', 'PI_70_30', 'PI_60_40'])

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
