 # AI Challenge HIT 2022 - Team 10. 
![](/assets/title.png)

-----------------------
 ## Table of Contents  
1.  [The Challenge](#thechallenge)  
2. [Repository Usage](#repousage)
    - [Depencencies](#dependencies) 
    - [Run the code](#runthecode) 

-----------------------


<a name="thechallenge"/>

## 1. [The Challenge](https://www.trentinoinnovation.eu/innova/strumenti-per-innovazione/public-ai-challenge/)
**MeteoTrentino** is a structure of the Autonomous
**Province of Trento** founded in 1997. It counts
more than 100 hundred weather stations all over
the Trentino region, collecting a variety of **meteorological data**:
- **temperature** 
- **humidity** 
- **wind intensity**
- **precipitation abundance**
- **snow levels**

The **data** collected from MeteoTrentino are
of interest not only for weather forecasting but also
for monitoring and research for example in *meteorology*, *nivology* and *glaciology*, therefore maintaining good quality archived data is of crucial importance.

The relevant number of weather station and the
critical conditions that these encounter (abundant
precipitations, critical temperatures, damage by animals and vegetation) make the task of validating
the data quite cumbersome. Damage, malfunctioning or anomalous conditions of the sensors may result in **untruthful data**. 

Intervening to restore the
sensors to proper functioning may consists in a remote reset but often it may require a **manual intervention** on the weather station site. Such intervention understandably take several **hours** or even **days**, therefore detecting anomalies in the shortest possible time is essential in order to garantee continuity of good quality stored data.

Currently, anomaly detection is performed
**“manually”** by an operator of MeteoTrentino,
based on a protocol which generates alerts
about which sensor/weather station encountered an
anomaly. At the moment, the alerts are generated
by a simple signal analysis code based on threshold
levels of the acquired signals. The process is overall too slow due to the **huge load of incoming data**
and the low performance of the code, which is often
not able to detect anomalies that remain unnoticed
for long times.


The **challenge** proposed by MeteoTrentino is to
use **Artificial Intelligence** to make up a code able to
effectively detect anomalies in the real-time data
and be integrated in the protocol to generate the
alerts, thus supporting the operator in charge of data quality control in his work.

---------------
<a name="repousage" />

## 2. Repository Usage

<a name="dependencies" />

- ## Dependecies

#### [Pytorch](https://pytorch.org/get-started/locally/)
```shell
# Python 3.x
pip3 install torch torchvision
```

#### [os-sys](https://pypi.org/project/os-sys/#files)
```shell
pip install os-sys
```


>Easiest way is to install the rest of the libraries/modules is [Anaconda](https://docs.continuum.io/anaconda) [[here](https://docs.continuum.io/anaconda/install/)]

otherwise 

#### [Numpy](https://numpy.org/install/)
```shell
pip install numpy
```
#### [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
```shell
python -m pip install -U matplotlib
```
#### [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
```shell
pip install pandas
```

----------

<a name="runthecode" />

- ## Run the code
clone the repo with
```shell
git clone https://github.com/eliazonta/AI-Challenge-HIT.git
```

- ## Structure
![](/CodeScheme.png)

<!-- ## Python code:
*python_code*  contiene:

* ` /python_code/notebooks/ ` per i notebooks
* ` /python_code/notebooks/DecisionTree ` codice con il test dell'*albero decisionale*
* ` /python_code/notebooks/AnomalyDetection ` codice per Anomaly Detection sui dati di una singola stazione
* ` /python_code/notebooks/TSF ` codice per il data splitting delle serie temporali
* ` /python_code/notebooks/converting script ` codice con il convertitore semi-automatico dei dataset

## how to run the notebooks on google colab: 
Clone the repository with its libraries
`
!git clone https://github.com/eliazonta/AI-Challenge-HIT
import sys
sys.path.append("/content/AI-Challenge-HIT/python_code/notebooks")
` 
-->


