# UQ_Surrogate_model
The UQ metrics are categorized in three parts: single time series, ensemble member, and multiple time series.

- For single time series, we consider moving-window based metrics instead of point to point metrics. These metrics reflect the informativeness of the given data.
- For ensemble members, we quantify uncertainty from multiple ensemble data sources.
- For multiple spatial time series, we quantify the uncertainty from wind farm turbine layout. The uncertainty among turbines is converted to a function of uncertainty and distance. UQ metric is summarized as follows:
<table>
    <thead>
        <tr>
            <th>   </th>
            <th>Type of Metric</th>
            <th>Metric</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>Single time series</td>
            <td rowspan=4>Moving-window</td>
            <td>Entropy</td>
        </tr>
        <tr>
            <td>Standard Deviation</td>
        </tr>
        <tr>
            <td>Turbulence Intensity</td>
        </tr>
        <tr>
            <td>Variability Index</td>
        </tr>
        <tr>
            <td rowspan=8>Ensemble member</td>
            <td rowspan=4>Moving-window</td>
        <td>Correlation</td>
        </tr>
        <tr>
            <td>MAPE</td>
        </tr>
        <tr>
            <td>nRMSE</td>
        </tr>
        <tr>
            <td>nMAE</td>
        </tr>
        <tr>
            <td rowspan=4>Distribution</td>
            <td>Spread Index</td>
        </tr>
        <tr>
            <td>Predictability Index</td>
        </tr>
        <tr>
            <td>CRPS</td>
        </tr>
        <tr>
            <td>Veritication rank histogram</td>
        </tr>
        <tr>
                <tr>
            <td rowspan=5>Multiple time series</td>
            <td rowspan=2>Moving-window</td>
        <td>Correlation</td>
        </tr>
        <tr>
            <td>MAPE</td>
        </tr>
        <tr>
            <td rowspan=2>Distribution</td>
            <td>Spread Index</td>
        </tr>
        <tr>
            <td>Predictability Index</td>
        </tr>
         <td rowspan=1>Spatial</td>
            <td>Kriging Variogram</td>
        </tr>
        <tr>
    </tbody>
</table>