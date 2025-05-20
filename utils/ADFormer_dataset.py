import os
import torch
import numpy as np
import pickle
import datetime
import geopandas as gpd
from utils.utils import StandardScaler, create_dataloader
from tqdm import tqdm
from fastdtw import fastdtw
from model.module import hierarchical_clustering

class ADFormerDataset():

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.interval = args.time_interval
        self.points_per_hour = 3600 // self.interval
        self.num_reg = args.num_reg

        if self.args.dataset_name == 'NYC-Taxi':
            self.data_files = ['./data/NYC_Taxi_origin.pkl', './data/NYC_Taxi_destination.pkl']
        elif self.args.dataset_name == 'NYC-Bike':
            self.data_files = ['./data/NYC_Bike_origin.pkl', './data/NYC_Bike_destination.pkl']
        elif self.args.dataset_name == 'XIAN-Taxi':
            self.data_files = ['./data/xian_Taxi_origin.pkl', './data/xian_Taxi_destination.pkl']

        self.get_matrixes()
        self.get_dtw()
        self.get_cluster()
        


    def get_data(self):
        cache_data_file_path = f'./cache/{self.args.dataset_name}_split_dataset_{self.args.window}_{self.args.horizon}.npz'
        # get split data
        if os.path.exists(cache_data_file_path):
            self.logger.info("Loading splited dataset from " + cache_data_file_path)
            cat_data = np.load(cache_data_file_path)
            x_train, y_train = cat_data['x_train'], cat_data['y_train']
            x_test, y_test = cat_data['x_test'], cat_data['y_test']
            x_val, y_val = cat_data['x_val'], cat_data['y_val']
            self.logger.info(f"train x: {x_train.shape}, y: {y_train.shape}")
            self.logger.info(f"val x: {x_val.shape}, y: {y_val.shape}")
            self.logger.info(f"test x: {x_test.shape}, y: {y_test.shape}")
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = self.generate_train_val_test()

        # scale the features, not external_feature
        self.output_dim = self.args.output_dim
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim

        self.scaler = StandardScaler(mean=x_train[..., :self.output_dim].mean(), std=x_train[..., :self.output_dim].std())
        self.logger.info(f"Standard Scaler mean: {self.scaler.mean}, std: {self.scaler.std}")
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])

        self.train_dataloader = create_dataloader(x_train, y_train, batch_size=self.args.batch_size, shuffle=True)
        self.val_dataloader = create_dataloader(x_val, y_val, batch_size=self.args.batch_size, shuffle=False)
        self.test_dataloader = create_dataloader(x_test, y_test, batch_size=self.args.batch_size, shuffle=False)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader
    

    def generate_train_val_test(self):
        x, y = self.generate_full_XY()
        test_rate = 1 - self.args.train_rate - self.args.val_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.args.train_rate)
        num_val = num_samples - num_test - num_train

        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        x_test, y_test = x[-num_test:], y[-num_test:]

        self.logger.info("Split train data according to date.")
        self.logger.info(f"train x: {x_train.shape}, y: {y_train.shape}")
        self.logger.info(f"val x: {x_val.shape}, y: {y_val.shape}")
        self.logger.info(f"test x: {x_test.shape}, y: {y_test.shape}")

        np.savez_compressed(
            f'./cache/{self.args.dataset_name}_split_dataset_{self.args.window}_{self.args.horizon}',
            x_train = x_train,
            y_train = y_train,
            x_val = x_val,
            y_val = y_val,
            x_test = x_test,
            y_test = y_test
        )
        self.logger.info(f"Saved the split dataset at: ./cache/{self.args.dataset_name}_split_dataset_{self.args.window}_{self.args.horizon}.")
        return x_train, y_train, x_val, y_val, x_test, y_test
    

    def generate_full_XY(self):
        # read od data
        for idx, filename in enumerate(self.data_files):
            if idx == 0:
                data = np.array(pickle.load(open(filename, 'rb'))).T
            else:
                data = np.stack((data, np.array(pickle.load(open(filename, 'rb'))).T), axis=-1)
        self.raw_data = data
        
        if self.args.load_external:
            # concat raw_X with the day of week and time of day.
            data = self.add_external_information(self.raw_data)
        x, y = self.generate_XY(data)
        self.logger.info("Added external_information to raw data and splited data according to the window and horizon.")
        self.logger.info(f"The entire dataset: x shape: {x.shape}, y shape: {y.shape}")
        return x, y

    def add_external_information(self, raw_data):
        num_samples, num_reg, feature_dim = raw_data.shape
    
        if self.args.dataset_name == 'NYC-Taxi':
            start_T = '2016-01-01T00:00:00'
            end_T = '2016-12-31T23:30:00'
        elif self.args.dataset_name == 'NYC-Bike':
            start_T = '2023-01-01T00:00:00'
            end_T = '2023-12-31T23:30:00'
        elif self.args.dataset_name == 'XIAN-Taxi':
            start_T = '2016-10-01T00:00:00'
            end_T = '2016-11-30T23:30:00'
        interval_seconds = self.interval
        start = np.datetime64(start_T)
        end = np.datetime64(end_T)
        num_slots = int((end - start) / np.timedelta64(interval_seconds, 's')) + 1
        timeslots = np.arange(start, end + np.timedelta64(interval_seconds, 's'), np.timedelta64(interval_seconds, 's'))

        data_list = [raw_data]
        time_ind = (timeslots - timeslots.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_reg, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)

        dayofweek = []
        for day in timeslots.astype("datetime64[D]"):
            dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
        day_in_week = np.zeros(shape=(num_samples, num_reg, 7))
        day_in_week[np.arange(num_samples), :, dayofweek] = 1
        data_list.append(day_in_week)

        data = np.concatenate(data_list, axis=-1)
        return data
    
    # split data according to the window and horizon
    def generate_XY(self, data):
        num_samples = data.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.args.window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.args.horizon + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y
    
    
    def get_matrixes(self):
        cache_matrix_file_path = './cache/' + self.args.dataset_city + '_matrixes.npz'
        if not os.path.exists(cache_matrix_file_path):
            self.logger.info('Generating  adj_mx/dist_mx  from zones.shp')
            NYC_gdf = gpd.read_file('./data/geo/taxi_zones/taxi_zones.shp')
            NYC_gdf = NYC_gdf.to_crs(epsg=32618)

            num_zones = len(set([locationID for locationID in NYC_gdf['LocationID']]))
            adj_mx = np.zeros((num_zones, num_zones), dtype=float)
            for i in range(num_zones):
                for j in range(i + 1, num_zones):
                    if NYC_gdf.iloc[i].geometry.touches(NYC_gdf.iloc[j].geometry):
                        adj_mx[i, j] = 1
                        adj_mx[j, i] = 1

            centroids = NYC_gdf.geometry.centroid
            coordinates = np.array([[point.x, point.y] for point in centroids])
            diff = np.abs(coordinates[:, None, :] - coordinates[None, :, :])    
            dist_mx = np.sum(diff, axis=2)

            np.savez_compressed(
                cache_matrix_file_path,
                adj_mx = adj_mx,
                dist_mx = dist_mx
            )
            
        self.logger.info(f'Loading  adj_mx/dist_mx  from {cache_matrix_file_path}.')
        matrixes = np.load(cache_matrix_file_path)
        self.adj_mx = matrixes['adj_mx']
        self.dist_mx = matrixes['dist_mx']
        
    
    def get_dtw(self):
        cache_path = './cache/' + self.args.dataset_name + '_dtwmx.npy'

        # create dtw matrix from data if not exist
        if not os.path.exists(cache_path):
            # read data
            file_names = self.data_files
            for idx, filename in enumerate(file_names):
                if idx == 0:
                    data = np.array(pickle.load(open(filename, 'rb'))).T
                else:
                    data = np.stack((data, np.array(pickle.load(open(filename, 'rb'))).T), axis=-1)

            data_mean = np.mean(
                [data[24 * self.points_per_hour * i: 24 * self.points_per_hour * (i + 1)]
                 for i in range(int(data.shape[0] * self.args.train_rate) // (24 * self.points_per_hour))], axis=0)
            dtw_distance = np.zeros((self.num_reg, self.num_reg))
            for i in tqdm(range(self.num_reg)):
                for j in range(i, self.num_reg):
                    dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
            for i in range(self.num_reg):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(cache_path, dtw_distance)

        self.dtw_mx = np.load(cache_path)
        self.logger.info('Load DTW matrix from {}'.format(cache_path))
        

    def get_cluster(self):
        cls_regs = eval(self.args.cluster_reg_nums)
        dtw_clusters = hierarchical_clustering(self.dtw_mx, cls_regs, self.args.bal_cls, self.args.cls_tolerance)
        dist_clusters = hierarchical_clustering(self.dist_mx, cls_regs, self.args.bal_cls, self.args.cls_tolerance)

        def get_map(clusters, reg_num, cluster_reg_nums):
            cluster_maps = []
            for i, cluster in enumerate(clusters):
                if i == 0:
                    map_matrix = torch.zeros((cluster_reg_nums[i], reg_num), dtype=torch.float32, device=self.args.device)
                else:
                    map_matrix = torch.zeros((cluster_reg_nums[i], cluster_reg_nums[i - 1]), dtype=torch.float32, device=self.args.device)
                
                for j, row in enumerate(cluster):
                    for idx in row:
                        map_matrix[j, idx] = 1

                if i == 0:
                    cur_map = map_matrix
                else:
                    cur_map = map_matrix @ cur_map
                    
                cluster_maps.append(cur_map)
            return cluster_maps

        self.dtw_map = get_map(dtw_clusters, self.args.num_reg, cls_regs)
        self.dist_map = get_map(dist_clusters, self.args.num_reg, cls_regs)
   

    # dataset properties
    def get_dataset_feature(self):
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, 
                "dist_mx": self.dist_mx, "dtw_mx": self.dtw_mx, 
                "dtw_map": self.dtw_map, "dist_map": self.dist_map,
                "ext_dim": self.ext_dim, "num_reg": self.num_reg, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim}
