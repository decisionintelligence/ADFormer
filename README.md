This repository provides the implementation code for our paper: 

*ADFormer: Aggregation Differential Transformer for Passenger Demand Forecasting*.
![The overall framework of ADFormer](./overall_framework.png)
The following is a detailed description of each folder.
## data
`geo/taxi_zones` contains files representing the geographic information of NYC, which can be used for map visualization. 

Similarly, `xian_hexAddr.txt` contains hexagonal grid addresses of Xi'an, which can be mapped to real-world geographic coordinates. 

Files such as `NYC_Taxi_origin.pkl` and `NYC_Taxi_destination.pkl` are stored as 2D arrays. The "origin/destination" refers to the start and end points of a trip, as the data are obtained from travel orders. The first dimension is the number of spatial units and the second is the number of timesteps. They are used to generate input data in `utils/ADFormer_dataset.py`.
## model
`module.py` implements the essential components required by *ADFormer*.

`ADFormer.py` forms the core of our method, implementing attention mechanisms across both spatial and temporal dimensions, as well as the ST-Encoder.
## utils
`ADFormer_config.py` is the configuration file used in our experiments, including settings for the dataset, model, and training process.

`ADFormer_dataset.py` serves several purposes:

(1) It processes raw data (eg. data/xxx_xxx_origin.pkl), such as adding external information and splitting it according to the window and horizon.

(2) It generates the adjacency, distance, and DTW distance matrices based on the geographic and raw data files.

(3) It aggregates spatial units using *dtw_mx* and produces cluster maps.

`ADFormer_trainer.py` defines the components and procedures for training, validation, and evaluation.

`utils.py` includes utility functions used in the experiments.

## Training
You can use the following command to train a model:
```
nohup python main.py --dataset_city XIAN --dataset_name XIAN-Taxi --window 6 --horizon 1 > XIAN-Taxi-6-1.log 2>&1 &
```
The trained model will be saved in the `log` folder.

Please note: Since reverse transformation may turn zeros into near-zero values (due to normalization using training data only), MAPE can be distorted. Thus, we compute MAPE only for values greater than 5.
