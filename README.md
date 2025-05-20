This repository provides the implementation code for our paper: 

*ADFormer: Aggregation Differential Transformer for Passenger Demand Forecasting*.
![The overall framework of ADFormer](./overall_framework.png)
The following is a detailed description of each folder.
## data
`geo/taxi_zones` contains files that indicate geographic information of NYC, which can be used to plot map. 

Similarly, `xian_hexAddr.txt` includes hexagonal grid addresses of Xi'an, which can be mapped to real-world latitude and longitude. 

These files, like `NYC_Taxi_origin.pkl` and `NYC_Taxi_destination.pkl`, are stored as 2D arraies. 'origin/destination' means 'the start point and end point of a trip', since we obtain these data from travel order. The first dimension is the number of spatial units and the second is the number of timesteps. They are used to generate input data in `utils/ADFormer_dataset.py`.
## model
`module.py` implements the essential components required bt *ADFormer*.

`ADFormer.py` constitutes the core part of our method, including the implementation of Attention from both spatial and temporal aspects, as well as the ST-Encoder.
## utils
`ADFormer_config.py` is the configuration file in our experiments and consists of settings of dataset, model and training process.

`ADFormer_dataset.py` serves several purposes:

(1) It processes raw data (eg. data/xxx_xxx_origin.pkl), such as adding external information to it, spliting it according to the window and horizon.

(2) We get *adjacent/distance/dtw_distance matrixes* there according to the geographic files and raw data files.

(3) We aggregate spatial units with *dtw_mx* and obtain the cluster maps.

`ADFormer_trainer.py` defines the components and process of training, validation and evaluation.

`utils.py` includes the utils needed by experiments.

## Training
You can use the following command to train a model:
```
nohup python main.py --dataset_city XIAN --dataset_name XIAN-Taxi --window 6 --horizon 1 > XIAN-Taxi-6-1.log 2>&1 &
```
The trained model will be saved in the `log` folder.
