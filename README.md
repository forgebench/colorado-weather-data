# Traffic and weather: a study of the impact of weather on traffic accident severity.
## All code, research, and writing done by Anthony Garcia.

### Abstract
This project is a study of the impact of weather on traffic accident severity. The goal was to use machine learning
algorithms to predict the severity of an accident based on the weather conditions that could be expressly entered from
weather predictions for the following day.

The data used in this project is from the
[Colorado Department of Transportation](https://www.codot.gov/safety/traffic-safety/data-analysis/crash-data) and the 
[National Oceanic and Atmospheric Administration](https://www.ncei.noaa.gov/). 

The data was collected from 2007 to 2020. The data was cleaned and analyzed using Python, SKLearn, and the Pandas 
library. The data was then visualized using Matplotlib. The data was then ran through machine learning algorithms from
SKLearn including RandomForest and Support Vector Machines in both regression and classification (multi and binary)
forms.

The results of this study show a near zero R2 correlation between the chosen weather statistics and the ability to
use machine learning to predict severity for the following day. This means that weather condition predictions, while
weather conditions may certainly have an impact on individual accidents, cannot be used to predict the severity of 
accidents for the following day as a whole.

### Data
The data used in this project is from the
[Colorado Department of Transportation](https://www.codot.gov/safety/traffic-safety/data-analysis/crash-data) and the 
[National Oceanic and Atmospheric Administration](https://www.ncei.noaa.gov/). 

The data was collected from 2007 to 2020. The Denver crash set was used for this project. The weather data was
collected from the Denver International Airport. These two sets of data were chosen due to the completeness of the data
as well as the fact that the weather data was collected from the same location as the crash data.

The data was analyzed using Python, SKLearn, and the Pandas library. The data was then visualized using Matplotlib where
needed. Scaling was done with SKLearn's StandardScaler and MinMaxScaler. Both were used to see which would work better.

Some data was removed from the set due to the fact that it was not needed for the analysis. This included the following:
dates, times, locations, and anything that was too sparse to be useful (such as the Weather Type columns) as well as
attribute flags.

What remained and was used for modeling is the following:
- Average Wind Speed
- Precipitation (rain)
- Snowfall
- Snow Depth
- Average Temperature
- Maximum Temperature
- Minimum Temperature
- Day of the week

Severity was calculated using the national standard for severity. This is a 1-5 scale where 1 is no injuries and 5
is a fatality. The severity was then calculated as follows for each accident:

Where y = number of passengers, x = injury scale number

Severity = y * (x * sqrt(x))

This calculation was repeated for each category of injury, then added together to get a severity score for that
particular accident. The severity scores for each accident on a given day were then compiled into a single score
which was used as the basis for the target variable.

The target variables were then further refined and multiple targets were tried. 

Tested targets were as follows:
- Raw severity score
- Raw accident numbers
- Severity score divided by accident numbers
- Severity score normalized by both MinMaxScaler and StandardScaler
- A binary target of 0 for no accidents and 1 for accidents using the severity score to separate the two (tried upper
- 50, 60, and 70 percentile as the cutoff)
- A multiple classification target splitting normalized severity into 5 categories (0-20, 20-40, 40-60, 60-80, 80-100)

### Machine Learning Algorithms, Methodology, and Results

The data was then ran through machine learning algorithms from SKLearn including RandomForest, Support Vector Machines,
and KNeighbors. It was run as both regression and classification (multi and binary) forms.

Multiple attempts at modeling were made and are documented in the analysis notebooks in dev-one and dev-two. The
results of these attempts are, in general, an R2 near zero with regression and a near 50% accuracy with classification.
Higher results of classification were seen when looked at with only the accuracy score, but a look at precision, recall,
and F1 score showed that the results were not as good as they seemed.

The results of the models are, unfortunately, showing that weather condition predictions cannot be used to predict
the severity of accidents for the following day as a whole.

### File Structure

The file structure of this project is as follows:
- data
    - data-dictionaries (contains data dictionaries for the data sets)
    - modified-data (contains the modified data sets)
    - raw-crash-data (contains the raw crash data sets)
    - raw-weather-data (contains the raw weather data sets)
- graphs (contains graphs generated by the analysis notebooks for top severities)
- notebooks
  - dev-one (contains the notebooks used for the first round of analysis)
  - dev-two (contains the notebooks used for the second round of analysis)

Please note that a lot of commentary has been left in the notebooks so if you wanted to follow along with this author's
train of thought then you may.


### Possible Reasons for Poor Results

It may be difficult to quantify changes in traffic patterns in snow versus other weather conditions. Snow will 
frequently cause more travelers to stay home thus lowering severity scores. It was hoped that the severity scores
divided into accidents could find this trend, but it was not found. It is noted that temperatures do seem to have a
noticeable R2 with the regression models, but this is not enough. 

Studies of weather and traffic accidents have been done, and it's clear that conditions like wet pavement and rain have
a [severe impact on traffic accidents](https://ops.fhwa.dot.gov/weather/q1_roadimpact.htm). This study also found that
the top 100 most severe accidents are clearly clustered in winter months. What is not clear is how we can use weather
predictions to predict the severity of accidents for the following day.

### Potential Future Work

A look at the crash data itself may be able to find trends that can be used to predict severity, since it includes
information about road conditions. Those conditions could then be extrapolated against weather data and a percentage
of accidents for each day in those conditions matching the weather data could be created. That number could then be
used to possibly predict severity.

### Conclusion

The results, as stated, show that weather condition predictions cannot be used to predict the severity of accidents for
the following day, at least with the dataset used. The dataset was chosen because of it's completeness and the fact that
the weather information could be gleaned from predictions. The conclusion is unfortunate, but the information gathered
is still helpful.
