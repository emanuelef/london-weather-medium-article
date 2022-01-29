## Introduction

This Notebook shows how to perform a quick analysis on simple time series by using basic Pandas and Seaborn commands to generate heatmaps.
The similar techniques can be used on any dataset containing just a date and value columns (number of sales, users accesses…).  
I hope the same steps can be useful when analyzing any time series dataset.

Let's start by importing the needed libraries, also setting retina resolution for plots and ggplot style.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import matplotlib.style as style
import missingno as msno

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

style.use('ggplot')
```

### Dataset  

I'm using a dataset downloaded from the National Centers for Environmental Information (NCEI), the data is in the public domain and can be used freely.  
In case using the same dataset or generating a new one from NCEI you need to cite the origin.  
The Dataset covers each day from 2010 to 2019 and the station used is located at Heathrow Airport in London.

**DATE**: is the year of the record (4 digits) followed by month (2 digits) and day (2 digits).  
**PRCP**: Precipitation (mm)  
**TAVG**: Average temperature (°C)  

```python
df = pd.read_csv('HeathrowMeteo2010-2019.csv', 
                               usecols=['DATE', 'PRCP', 'TAVG'], parse_dates=['DATE'])
#df['DATE'] = df['DATE'].astype('datetime64[ns]') # needed if date format is not standard
df.columns = ['date', 'precipitation', 'avg_temp']
df.sample(3)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>precipitation</th>
      <th>avg_temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2111</th>
      <td>2015-11-03</td>
      <td>0.3</td>
      <td>11.4</td>
    </tr>
    <tr>
      <th>3585</th>
      <td>2019-11-26</td>
      <td>1.8</td>
      <td>11.4</td>
    </tr>
    <tr>
      <th>1535</th>
      <td>2014-03-16</td>
      <td>0.0</td>
      <td>12.7</td>
    </tr>
  </tbody>
</table>
</div>


```python
df.dtypes
```




    date             datetime64[ns]
    precipitation           float64
    avg_temp                float64
    dtype: object



```python
len(df)
```




    3621



Let's check if there are missing values

```python
df.isnull().sum()
```




    date              0
    precipitation    30
    avg_temp          0
    dtype: int64



```python
round(df.isnull().mean() * 100, 2)
```




    date             0.00
    precipitation    0.83
    avg_temp         0.00
    dtype: float64



```python
_ = msno.matrix(df)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_10_0.png){: .center-image }

In this case we will fill the null values with 0.0 (in this case meaning no rain), this is to make the rest of the analysis simpler (like when summing up) but this may change depending on your dataset.

```python
df['precipitation'].fillna(0, inplace=True)
```

We also want to check if there are missing days in the range

```python
print(f"Data Available from {df.date.min()} to {df.date.max()}")
```

    Data Available from 2010-01-01 00:00:00 to 2019-12-31 00:00:00

```python
idx = pd.date_range(df.date.min(), df.date.max())
print(f"Days present {len(df)} out of {len(idx)}")
```

    Days present 3621 out of 3652

Some days are missing, let's create a DataFrame to visualize them.

```python
pd.DataFrame(data=idx.difference(df.date), columns=['dates']).sample(3)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>2015-04-11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-04-09</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2018-11-22</td>
    </tr>
  </tbody>
</table>
</div>


According to the MET Office a day can be considered dry if precipitation is less than 1mm.  
Let's see what is the percentage of dry days in the entire dataset.

```python
MIN_PRECIPITATION_MM_DRY = 1.0
```

```python
round((len(df[df['precipitation'] < MIN_PRECIPITATION_MM_DRY]) / len(df)) * 100, 2)
```




    70.81



Find day with the highest precpitation.

```python
df[df.precipitation == df.precipitation.max()][['date', 'precipitation']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>precipitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2043</th>
      <td>2015-08-27</td>
      <td>48.0</td>
    </tr>
  </tbody>
</table>
</div>


And highest temperature

```python
df[df.avg_temp == df.avg_temp.max()][['date', 'avg_temp']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>avg_temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3461</th>
      <td>2019-07-25</td>
      <td>28.6</td>
    </tr>
  </tbody>
</table>
</div>


```python
sns.distplot(df.precipitation)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x111506400>



![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_25_1.png){: .center-image }

In order to create the needed visualizations it can be useful to augment the dataframe with additional columns representing date infos.  
This step is not strictly needed because we could just use the same methods when later grouping.

```python
df['month'] = df.date.dt.month
df['year'] = df.date.dt.year
df['day'] = df.date.dt.day
df['weekdayName'] = df.date.dt.day_name() # df.date.dt.weekday_name on older Pandas
df['weekday'] = df.date.dt.weekday
df['week'] = df.date.dt.week
df['weekend'] = df.date.dt.weekday // 5 == 1
```

```python
df['raining'] = df['precipitation'].gt(MIN_PRECIPITATION_MM_DRY).astype('int')
```

```python
df.sample(3)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>precipitation</th>
      <th>avg_temp</th>
      <th>month</th>
      <th>year</th>
      <th>day</th>
      <th>weekdayName</th>
      <th>weekday</th>
      <th>week</th>
      <th>weekend</th>
      <th>raining</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1678</th>
      <td>2014-08-06</td>
      <td>9.7</td>
      <td>20.3</td>
      <td>8</td>
      <td>2014</td>
      <td>6</td>
      <td>Wednesday</td>
      <td>2</td>
      <td>32</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>2013-06-17</td>
      <td>0.3</td>
      <td>16.8</td>
      <td>6</td>
      <td>2013</td>
      <td>17</td>
      <td>Monday</td>
      <td>0</td>
      <td>25</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2320</th>
      <td>2016-05-30</td>
      <td>0.0</td>
      <td>14.1</td>
      <td>5</td>
      <td>2016</td>
      <td>30</td>
      <td>Monday</td>
      <td>0</td>
      <td>22</td>
      <td>False</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


We could create a heatmap representing the average precipitation for every month in the dataset.  
To do so we can use the pivot_table function:

```python
all_month_year_df = pd.pivot_table(df, values="precipitation",index=["month"],
                                   columns=["year"],
                                   fill_value=0,
                                   margins=True)
named_index = [[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_df.index)]]
all_month_year_df = all_month_year_df.set_index(named_index)
all_month_year_df
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>All</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan</th>
      <td>1.687097</td>
      <td>2.506452</td>
      <td>0.996774</td>
      <td>1.309677</td>
      <td>4.783871</td>
      <td>1.964516</td>
      <td>2.383871</td>
      <td>1.867742</td>
      <td>1.329032</td>
      <td>0.929032</td>
      <td>1.975806</td>
    </tr>
    <tr>
      <th>Feb</th>
      <td>3.196429</td>
      <td>1.535714</td>
      <td>0.579310</td>
      <td>1.389286</td>
      <td>3.671429</td>
      <td>1.457143</td>
      <td>1.572414</td>
      <td>1.450000</td>
      <td>0.989286</td>
      <td>1.346429</td>
      <td>1.714184</td>
    </tr>
    <tr>
      <th>Mar</th>
      <td>1.641935</td>
      <td>0.464516</td>
      <td>0.561290</td>
      <td>1.554839</td>
      <td>1.041935</td>
      <td>0.835484</td>
      <td>2.367742</td>
      <td>0.887097</td>
      <td>2.680645</td>
      <td>1.654839</td>
      <td>1.369032</td>
    </tr>
    <tr>
      <th>Apr</th>
      <td>0.726667</td>
      <td>0.086667</td>
      <td>2.926667</td>
      <td>1.130000</td>
      <td>1.880000</td>
      <td>0.922222</td>
      <td>1.593333</td>
      <td>0.123333</td>
      <td>2.156667</td>
      <td>0.426667</td>
      <td>1.217921</td>
    </tr>
    <tr>
      <th>May</th>
      <td>0.764516</td>
      <td>0.806452</td>
      <td>0.835484</td>
      <td>1.332258</td>
      <td>2.841935</td>
      <td>0.819355</td>
      <td>1.567742</td>
      <td>2.151613</td>
      <td>1.964516</td>
      <td>1.337037</td>
      <td>1.443464</td>
    </tr>
    <tr>
      <th>Jun</th>
      <td>0.250000</td>
      <td>2.643333</td>
      <td>3.390000</td>
      <td>0.400000</td>
      <td>1.356667</td>
      <td>0.423333</td>
      <td>3.513333</td>
      <td>1.546667</td>
      <td>0.016667</td>
      <td>2.736667</td>
      <td>1.627667</td>
    </tr>
    <tr>
      <th>Jul</th>
      <td>0.548387</td>
      <td>1.561290</td>
      <td>1.835484</td>
      <td>0.819355</td>
      <td>1.622581</td>
      <td>2.325806</td>
      <td>0.541935</td>
      <td>2.932258</td>
      <td>0.483871</td>
      <td>1.632258</td>
      <td>1.430323</td>
    </tr>
    <tr>
      <th>Aug</th>
      <td>3.019355</td>
      <td>2.154839</td>
      <td>1.141935</td>
      <td>1.067742</td>
      <td>3.174194</td>
      <td>3.412903</td>
      <td>0.700000</td>
      <td>1.858065</td>
      <td>1.538710</td>
      <td>1.100000</td>
      <td>1.916774</td>
    </tr>
    <tr>
      <th>Sep</th>
      <td>1.033333</td>
      <td>1.160000</td>
      <td>1.230000</td>
      <td>1.676667</td>
      <td>0.360000</td>
      <td>2.056667</td>
      <td>1.316667</td>
      <td>1.946667</td>
      <td>0.563333</td>
      <td>1.856667</td>
      <td>1.320000</td>
    </tr>
    <tr>
      <th>Oct</th>
      <td>2.000000</td>
      <td>0.409677</td>
      <td>2.454839</td>
      <td>1.596774</td>
      <td>2.480645</td>
      <td>1.296774</td>
      <td>0.800000</td>
      <td>0.448387</td>
      <td>1.845161</td>
      <td>3.125806</td>
      <td>1.645806</td>
    </tr>
    <tr>
      <th>Nov</th>
      <td>0.830000</td>
      <td>0.896667</td>
      <td>2.180000</td>
      <td>1.676667</td>
      <td>4.270000</td>
      <td>1.573333</td>
      <td>2.900000</td>
      <td>1.160000</td>
      <td>2.575000</td>
      <td>2.616667</td>
      <td>2.057483</td>
    </tr>
    <tr>
      <th>Dec</th>
      <td>0.629032</td>
      <td>1.829032</td>
      <td>2.709677</td>
      <td>2.993548</td>
      <td>1.235484</td>
      <td>1.258065</td>
      <td>0.341935</td>
      <td>2.477419</td>
      <td>1.890323</td>
      <td>2.912903</td>
      <td>1.827742</td>
    </tr>
    <tr>
      <th>All</th>
      <td>1.352603</td>
      <td>1.337808</td>
      <td>1.735519</td>
      <td>1.414521</td>
      <td>2.387397</td>
      <td>1.569767</td>
      <td>1.625956</td>
      <td>1.575890</td>
      <td>1.491086</td>
      <td>1.814127</td>
      <td>1.630875</td>
    </tr>
  </tbody>
</table>
</div>


```python
def plot_heatmap(df, title):
    plt.figure(figsize = (14, 10))
    ax = sns.heatmap(df, cmap='RdYlGn_r',
                     robust=True,
                     fmt='.2f', annot=True,
                     linewidths=.5, annot_kws={'size':11},
                     cbar_kws={'shrink':.8, 'label':'Precipitation (mm)'})
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    plt.title(title, fontdict={'fontsize':18}, pad=14);
```

robust or vmin, vmax

```python
plot_heatmap(all_month_year_df, 'Average Precipitations')
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_34_0.png){: .center-image }

By changing the aggfunc on pivot_table creation we can have a different aggregation, for instance we can use aggfunc=np.sum to calculate the total amount of rainfall per month.

```python
all_month_year_sum_df = pd.pivot_table(df, values="precipitation",index=["month"], columns=["year"], aggfunc=np.sum, fill_value=0)
all_month_year_sum_df = all_month_year_sum_df.set_index([[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_sum_df.index)]])
plot_heatmap(all_month_year_sum_df, 'Total Precipitations')
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_36_0.png){: .center-image }

And if we wanted to calculate the average amount of rainfall per weekday we just need to change index.

```python
all_weekday_year_df = pd.pivot_table(df, values="precipitation",index=["weekday"], columns=["year"], fill_value=0.0)
all_weekday_year_df = all_weekday_year_df.set_index([[calendar.day_name[i] for i in list(all_weekday_year_df.index)]])
plot_heatmap(all_weekday_year_df, 'Average Precipitation per weekday')
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_38_0.png){: .center-image }

The aggfunc can be a custom function, it could be interesting to calculate the percentage of days in the month with rain.

```python
all_month_year_percentage_df = pd.pivot_table(df, values="precipitation",index=["month"], columns=["year"],
                                              aggfunc=lambda x: (x>MIN_PRECIPITATION_MM_DRY).sum()/len(x),
                                              fill_value=0,
                                              margins=True)
all_month_year_percentage_df = all_month_year_percentage_df.set_index([[calendar.month_abbr[i] if isinstance(i, int)
                                                                        else i for i in list(all_month_year_percentage_df.index)]])
```

```python
plt.figure(figsize = (14, 10))
ax = sns.heatmap(all_month_year_percentage_df, cmap = 'RdYlGn_r', annot=True, fmt='.0%',
                 vmin=0, vmax=1, linewidths=.5, annot_kws={"size": 16})
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, .25, .50,.75, 1])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 14)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 14)
ax.tick_params(rotation = 0)
plt.title('Percentage of days in the month with rain', fontdict={'fontsize':18}, pad=14);
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_41_0.png){: .center-image }

In a similar fashion we can create an aggregation for each day of the year.

```python
def plot_heatmap_year(year):
    plt.figure(figsize = (16, 10))
    allByYear_df = df.loc[df['year'] == year]
    allByYear_df = pd.pivot_table(allByYear_df, values="precipitation",
                                  index=["month"], columns=["day"], fill_value=None)
    allByYear_df = allByYear_df.set_index([[calendar.month_abbr[i] for i in list(allByYear_df.index)]])
    ax = sns.heatmap(allByYear_df, cmap = 'RdYlGn_r',
                     vmin=0, vmax=20,
                     annot=False, linewidths=.1,
                     annot_kws={"size": 8}, square=True, cbar_kws={"shrink": .48, 'label': 'Rain (mm)'})
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)
    ax.tick_params(rotation = 0)
    plt.title(f'Precipitations {year}', fontdict={'fontsize':18}, pad=14);
```

```python
plot_heatmap_year(2019)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_44_0.png){: .center-image }

```python
plot_heatmap_year(2014)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_45_0.png){: .center-image }

```python
all_days_avg_df = df.groupby([df.date.dt.month, df.date.dt.day])['precipitation'].mean()
all_days_avg_df = all_days_avg_df.unstack()
all_days_avg_df = all_days_avg_df.set_index([[calendar.month_abbr[i] for i in list(all_days_avg_df.index)]])
```

Another possibility is to get the mean precipitation for each day of the year considering all the years in the dataset.
In the heatmap a custom colormap was used in order to have greenish cells for very low values.

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_colormap(seq):
    """
    Return a LinearSegmentedColormap
    seq: list
        a sequence of floats and RGB-tuples. 
        The floats should be increasing and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
```

```python
import matplotlib.colors as colors
c = colors.ColorConverter().to_rgb
gyr = make_colormap([c('green'), c('yellow'), 0.25, c('yellow'), c('red')])
```

```python
plt.figure(figsize = (20, 14))
ax = sns.heatmap(all_days_avg_df, cmap = gyr, annot=True, fmt='.2f',
                 vmin=0, linewidths=.1,
                 annot_kws={"size": 8}, square=True,  # <-- square cell
                 cbar_kws={"shrink": .5, 'label': 'Rain (mm)'})
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)
ax.tick_params(rotation = 0)
_ = plt.title('Precipitations Average 2010-2019', fontdict={'fontsize':18}, pad=14)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_50_0.png){: .center-image }

Another way to customize the colormap to make highest values more prominent.

```python
custom_palette = sns.color_palette("GnBu", 6)
custom_palette[5] = sns.color_palette("OrRd", 6)[5]
```

```python
sns.palplot(custom_palette)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_53_0.png){: .center-image }

```python
plt.figure(figsize = (20, 14))
ax = sns.heatmap(all_days_avg_df, cmap = custom_palette, annot=True, fmt='.2f',
                 vmin=0, linewidths=.1,
                 annot_kws={"size": 8}, square=True,
                 cbar_kws={"shrink": .5, 'label': 'Rain (mm)'})
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)
ax.tick_params(rotation = 0)
_ = plt.title('Precipitations Average 2010-2019', fontdict={'fontsize':18}, pad=14)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_54_0.png){: .center-image }

### CountPlot with customized text

```python
df.groupby('year')['raining'].sum()
```




    year
    2010     88
    2011     74
    2012    101
    2013     88
    2014    134
    2015    101
    2016    100
    2017     94
    2018     91
    2019    124
    Name: raining, dtype: int64



```python
plt.figure(figsize = (14, 6))
ax = sns.countplot(x="year", hue="raining", data=df.sort_values(by='year'))
ax.legend(loc='upper right', frameon=True, labels=['Dry', 'Rain'])

for p in ax.patches:
    ax.annotate(format(p.get_height()),
                (p.get_x()+p.get_width()/2., p.get_height()-4),
                ha = 'center', va = 'center',
                xytext = (0, 10), textcoords = 'offset points')

_ = ax.set_title("Dry and Wet Days per Year")
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_57_0.png){: .center-image }

Something that may be needed when analyzing time series is to calculate the number of consecutive days satisfying a specific condition.
In this case we can try to find the longest spells of dry and rainy days.
One way to do so is to use a combination of diff, cumsum and groupby.
First we need to label each row with an increasing number per each spell.

```python
df['value_grp'] = (df['raining'].diff() != 0).astype('int').cumsum()
```

```python
(df['raining'].diff() != 0).astype('int')
```




    0       1
    1       0
    2       0
    3       0
    4       0
           ..
    3616    1
    3617    1
    3618    0
    3619    0
    3620    0
    Name: raining, Length: 3621, dtype: int64



```python
df.head(10)[['date', 'precipitation', 'raining', 'value_grp']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>precipitation</th>
      <th>raining</th>
      <th>value_grp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-01-01</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-01-02</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-01-03</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-01-04</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-01-05</td>
      <td>0.3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2010-01-06</td>
      <td>1.5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2010-01-07</td>
      <td>3.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2010-01-08</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2010-01-09</td>
      <td>0.3</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2010-01-10</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


Then we can create a new DataFrame with a row for each label using groupby.

```python
grouped_values = df.groupby('value_grp')
consecutive_df = pd.DataFrame({'BeginDate' : grouped_values.date.first(), 
              'EndDate' : grouped_values.date.last(),
              'Consecutive' : grouped_values.size(),
              'condition': grouped_values.raining.max() }).reset_index(drop=True)
consecutive_df['condition'].replace({0: 'Dry', 1: 'Rain'}, inplace=True)
consecutive_df.sort_values(by='Consecutive', ascending=False).head(10)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BeginDate</th>
      <th>EndDate</th>
      <th>Consecutive</th>
      <th>condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>974</th>
      <td>2018-05-31</td>
      <td>2018-07-27</td>
      <td>58</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2010-06-10</td>
      <td>2010-07-14</td>
      <td>35</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>836</th>
      <td>2017-03-24</td>
      <td>2017-04-26</td>
      <td>34</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2011-04-06</td>
      <td>2011-05-06</td>
      <td>31</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>196</th>
      <td>2011-11-06</td>
      <td>2011-11-30</td>
      <td>25</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>114</th>
      <td>2011-01-19</td>
      <td>2011-02-10</td>
      <td>23</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>1026</th>
      <td>2018-12-25</td>
      <td>2019-01-16</td>
      <td>23</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>432</th>
      <td>2013-11-22</td>
      <td>2013-12-13</td>
      <td>22</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>2019-04-11</td>
      <td>2019-05-02</td>
      <td>22</td>
      <td>Dry</td>
    </tr>
    <tr>
      <th>388</th>
      <td>2013-07-04</td>
      <td>2013-07-24</td>
      <td>21</td>
      <td>Dry</td>
    </tr>
  </tbody>
</table>
</div>


```python
plt.figure(figsize = (14, 6))
ax = sns.countplot(x='Consecutive', hue='condition', data=consecutive_df.query('Consecutive >= 2'))
ax.set_title('Consecutive days on a specific condition 2012-2019 (> 2 days)', pad=14)
ax.set(xlabel='Consecutive days', ylabel='Count')
_ = plt.legend(loc='upper right')
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_64_0.png){: .center-image }

```python
consecutive_df['DateRange'] = consecutive_df["BeginDate"].astype(str) + ' -> ' + consecutive_df["EndDate"].astype(str)
ax = sns.barplot(x="Consecutive", y="DateRange", hue="condition", data=consecutive_df.sort_values(by='Consecutive', ascending=False).head(14))

for p in ax.patches:
 width = p.get_width()
 ax.text(width -1.6, p.get_y() + p.get_height()/2. + 0.2,'{:1.0f}'.format(width), ha="center")
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_65_0.png){: .center-image }

```python
df_top10_per_condition = consecutive_df.sort_values(by='Consecutive',ascending = False).groupby('condition').head(10)

d = {'color': ['g', 'r']}
g = sns.FacetGrid(df_top10_per_condition, row="condition",
                      hue='condition',
                      hue_kws=d,
                      sharey=False)

g.fig.set_figheight(8)
g.fig.set_figwidth(10)
    
_ = g.map(sns.barplot, "Consecutive", "DateRange")
_ = g.set(ylabel='')

# This is just to add the numbers inside the bars
for ax in g.axes.flat:
 for p in ax.patches:
  width = p.get_width()
  _ = ax.text(width -1.6, p.get_y() + p.get_height()/2. + 0.1,'{:1.0f}'.format(width), ha="center")
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_66_0.png){: .center-image }

Create custom palette to use blue for high precipitation values

```python
custom_palette = sns.diverging_palette(128, 240, n=10)
```

```python
sns.palplot(custom_palette)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_69_0.png){: .center-image }

```python
plt.figure(figsize = (20, 14))
ax = sns.heatmap(all_days_avg_df, cmap = custom_palette, annot=True, fmt='.2f',
                 vmin=0, linewidths=.1,
                 annot_kws={"size": 8}, square=True,
                 cbar_kws={"shrink": .5, 'label': 'Rain (mm)'})
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)
ax.tick_params(rotation = 0)
_ = plt.title('Precipitations Average 2010-2019', fontdict={'fontsize':18}, pad=14)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_70_0.png){: .center-image }

Running average to identify periods with high precipitations.

```python
plt.figure(figsize = (18, 6))
plt.title('Avg Rainfall (40 days window)', pad=14)
_ = df.set_index('date')['precipitation'].rolling(40).mean().plot()
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_72_0.png){: .center-image }

Experimenting with the average temperature

```python
ops_month_df= df.groupby(['month', 'year']).mean()['avg_temp'].reset_index()
plt.figure(figsize = (14, 6))
ax = sns.boxplot(x = "month", y = "avg_temp", data = ops_month_df)
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_74_0.png){: .center-image }

```python
df.groupby(['month', 'year']).mean()['avg_temp'].reset_index()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>year</th>
      <th>avg_temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2010</td>
      <td>2.183871</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2011</td>
      <td>5.083871</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2012</td>
      <td>6.593548</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2013</td>
      <td>4.551613</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2014</td>
      <td>7.006452</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>12</td>
      <td>2015</td>
      <td>11.374194</td>
    </tr>
    <tr>
      <th>116</th>
      <td>12</td>
      <td>2016</td>
      <td>6.883871</td>
    </tr>
    <tr>
      <th>117</th>
      <td>12</td>
      <td>2017</td>
      <td>5.990323</td>
    </tr>
    <tr>
      <th>118</th>
      <td>12</td>
      <td>2018</td>
      <td>7.941935</td>
    </tr>
    <tr>
      <th>119</th>
      <td>12</td>
      <td>2019</td>
      <td>7.077419</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 3 columns</p>
</div>


```python
def plotHeatmap(df, title):
    plt.figure(figsize = (20, 8))

    ax = sns.heatmap(df, cmap = 'RdYlBu_r', fmt='.2f', annot=True,
                     linewidths=.2, annot_kws={"size": 8}, square=True,
                     cbar_kws={"shrink": .9, 'label': 'Temperature °C'})
    cbar = ax.collections[0].colorbar
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 10)
    ax.tick_params(rotation = 0)
    plt.title(title, fontdict={'fontsize':18}, pad=14);
```

```python
allMonthYear_df = pd.pivot_table(df, values="avg_temp",index=["month"], columns=["year"], fill_value=None, margins=True)
allMonthYear_df = allMonthYear_df.set_index([[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(allMonthYear_df.index)]])
plotHeatmap(allMonthYear_df, 'Average Temperature')
```

![png]({{ site.baseurl }}/assets/images/EDA%20on%20Meteo%20Data/EDA%20on%20Meteo%20Data_77_0.png){: .center-image }

```python

```
