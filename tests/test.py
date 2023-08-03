#--------------------------------  NOTE  ----------------------------------------
# 1 This code is to quantify uncertainty based on multiple time series;
# 2 The format of the input data is [ts ts1 ...tsN];
# 3 Coder: Honglin Li        Date: 08/03/2023       @ UT-Dallas
#--------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.signal import spectral_entropy
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as nrmse
import antropy as ant

class UncertaintyMetrics:
    """
    Base class for uncertainty qualification metrics.
    """

    def __init__(self, data):
        """
        Constructor for the UncertaintyMetrics class.

        Parameters:
        - data (dict): A dictionary containing the data for the metrics calculation.
        """
        self.data = data

    def calculate_metrics(self):
        """
        Abstract method to calculate the uncertainty metrics.
        """
        pass

    def print_metrics(self):
        """
        Abstract method to print the calculated metrics.
        """
        pass


class SingleTimeSeriesMovingWindowMetrics(UncertaintyMetrics):
    """
    Subclass for single time series moving-window metrics.
    """

    def __init__(self, data, window_size):
        """
        Constructor for the SingleTimeSeriesMovingWindowMetrics class.

        Parameters:
        - data (dict): A dictionary containing the data for the metrics calculation.
        - window_size (int): The size of the moving window for calculation.
        """
        super().__init__(data)
        self.window_size = window_size

    def calculate_entropy(self, window):
        return ant.spectral_entropy(window, sf=100, method='welch', normalize=True)
    def entropy(self):
        """
        Calculate Spectral Entropy for each window.

        Returns:
            pandas.DataFrame: DataFrame containing Spectral Entropy values for each window.
        """
        entropy_df = pd.DataFrame(columns=['Entropy'])
        for i in range(len(self.data) - self.window_size):
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
        for i in range(len(self.data) - self.window_size):
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
        for i in range(len(self.data) - self.window_size):
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
        for i in range(len(self.data) - self.window_size):
            window = self.data[i:i + self.window_size]
            result = (np.max(window) - np.min(window)) / (np.mean(window) + 0.001)
            var_df.loc[i] = result
        return var_df

    def calculate_metrics(self):
        """
        Calculate the single time series moving-window metrics.

        Returns:
        - dict: A dictionary containing the calculated metrics.
        """
        en_era = []
        en_wtk = []
        sd_era = []
        sd_wtk = []
        ti_era = []
        ti_wtk = []
        vi_era = []
        vi_wtk = []

        for i in range(len(self.data) - self.window_size + 1):
            window_era = self.data['ERA'][i:i + self.window_size]
            window_wtk = self.data['WTK'][i:i + self.window_size]

            en_era.append(spectral_entropy(window_era, method='fft'))
            en_wtk.append(spectral_entropy(window_wtk, method='fft'))

            sd_era.append(np.std(window_era))
            sd_wtk.append(np.std(window_wtk))

            ti_era.append((np.max(window_era) - np.min(window_era)) / np.mean(window_era))
            ti_wtk.append((np.max(window_wtk) - np.min(window_wtk)) / np.mean(window_wtk))

            vi_era.append(np.std(window_era) / np.mean(window_era))
            vi_wtk.append(np.std(window_wtk) / np.mean(window_wtk))

        result = {
            'entropy': np.column_stack((en_era, en_wtk)),
            'sd': np.column_stack((sd_era, sd_wtk)),
            'ti': np.column_stack((ti_era, ti_wtk)),
            'vi': np.column_stack((vi_era, vi_wtk))
        }

        return result

    def print_metrics(self):
        """
        Print the single time series moving-window metrics.
        """
        print("Single Time Series Moving Window Metrics:")
        # Implement the method to print the calculated metrics
        pass


class EnsembleMember(UncertaintyMetrics):
    """
    Subclass for ensemble member metrics.
    """

    def __init__(self, data, ensemble_type):
        """
        Constructor for the EnsembleMember class.

        Parameters:
        - data (dict): A dictionary containing the data for the metrics calculation.
        - ensemble_type (str): The type of ensemble member metrics to calculate (e.g., 'MovingWindow', 'Distribution').
        """
        super().__init__(data)
        self.ensemble_type = ensemble_type

    def calculate_metrics(self):
        """
        Calculate the ensemble member metrics based on the specified type.

        Returns:
        - dict: A dictionary containing the calculated metrics.
        """
        if self.ensemble_type == 'MovingWindow':
            cor_ens = []
            mape_ens = []
            nrmse_ens = []
            nmae_ens = []

            for i in range(len(self.data) - 12):
                ens1 = self.data['ens1'][i:i + 12]
                ensN = self.data['ensN'][i:i + 12]

                cor_ens.append(np.corrcoef(ens1, ensN)[0, 1])
                mape_ens.append(MAPE(ens1, ensN))
                nrmse_ens.append(nrmse(ens1, ensN, squared=False))
                nmae_ens.append(np.mean(np.abs(ens1 - ensN)) / 12)

            result = {
                'correlation': cor_ens,
                'MAPE': mape_ens,
                'nRMSE': nrmse_ens,
                'nMAE': nmae_ens
            }

            return result

        elif self.ensemble_type == 'Distribution':
            # Implement the R code for ensemble member distribution metrics in Python
            pass

    def print_metrics(self):
        """
        Print the ensemble member metrics.
        """
        print("Ensemble Member Metrics:")
        # Implement the method to print the calculated metrics
        pass


class MultipleTimeSeries(UncertaintyMetrics):
    """
    Subclass for multiple time series metrics.
    """

    def __init__(self, data, series_type):
        """
        Constructor for the MultipleTimeSeries class.

        Parameters:
        - data (dict): A dictionary containing the data for the metrics calculation.
        - series_type (str): The type of multiple time series metrics to calculate (e.g., 'MovingWindow', 'Distribution', 'Spatial').
        """
        super().__init__(data)
        self.series_type = series_type

    def calculate_metrics(self):
        """
        Calculate the multiple time series metrics based on the specified type.

        Returns:
        - dict: A dictionary containing the calculated metrics.
        """
        if self.series_type == 'MovingWindow':
            cor_ts = []
            mape_ts = []

            for i in range(len(self.data) - 12):
                ts1 = self.data['ts1'][i:i + 12]
                tsN = self.data['tsN'][i:i + 12]

                cor_ts.append(np.corrcoef(ts1, tsN)[0, 1])
                mape_ts.append(MAPE(ts1, tsN))

            result = {
                'correlation': cor_ts,
                'MAPE': mape_ts,
            }

            return result

        elif self.series_type == 'Distribution':
            # Implement the R code for multiple time series distribution metrics in Python
            pass

        elif self.series_type == 'Spatial':
            # Implement the R code for spatial metrics in Python
            pass

    def print_metrics(self):
        """
        Print the multiple time series metrics.
        """
        print("Multiple Time Series Metrics:")
        # Implement the method to print the calculated metrics
        pass


class Spatial(UncertaintyMetrics):
    """
    Subclass for spatial metrics.
    """

    def __init__(self, data):
        """
        Constructor for the Spatial class.

        Parameters:
        - data (dict): A dictionary containing the data for the metrics calculation.
        """
        super().__init__(data)

    def calculate_metrics(self):
        """
        Calculate the spatial metrics.

        Returns:
        - dict: A dictionary containing the calculated metrics.
        """
        nug = []
        ran = []
        sill = []

        # Implement the R code for spatial metrics in Python

        result = {
            'nugget': nug,
            'range': ran,
            'sill': sill,
        }

        return result

    def print_metrics(self):
        """
        Print the spatial metrics.
        """
        print("Spatial Metrics:")
        # Implement the method to print the calculated metrics
        pass
