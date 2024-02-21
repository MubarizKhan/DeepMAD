import pandas as pd
import numpy as np
from rich.theme import Theme
from rich.console import Console

# dict of rich colors
# color used in project
ct = Theme({
    'good': "bold green ",
    'bad': "red",
    'blue': "blue",
    'yellow': "yellow",
    'purple': "purple",
    'magenta': "magenta",
    'cyan': "cyan"
})
rc = Console(record=True, theme=ct)


class Preprocessing:

    def __init__(self,df):
        self.df = df        
        # self.df = self.r_hdf()
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        # return self.df

    def rm_col(self):
        del self.df['src_port']
        del self.df['mean_bpktl']
        del self.df['bpsh_cnt']
        del self.df['total_bpktl']
        del self.df['mean_active_s']
        del self.df['max_active_s']
        del self.df['downUpRatio']
        del self.df['flow']
        del self.df['src']
        del self.df['dst']
        del self.df['protocol']
        del self.df['timestamp']
        del self.df['std_biat']
        del self.df['furg_cnt']
        del self.df['burg_cnt']
        del self.df['total_bhlen']
        del self.df['flow_cwr']
        del self.df['flow_ece']
        del self.df['std_active_s']
        del self.df['min_active_s']
        del self.df['fAvgBytesPerBulk']
        del self.df['fAvgPacketsPerBulk']
        del self.df['bAvgPacketsPerBulk']
        del self.df['fAvgBulkRate']
        del self.df['bAvgBytesPerBulk']

        del self.df['bAvgBulkRate']
        del self.df['mean_biat']
        del self.df['min_biat']
        del self.df['label']
        del self.df['bInitWinSize']
        # del self.df['flow_syn']
        del self.df['flow_rst']
        return self.df

    def r_csv(self, filename):
        df = pd.read_csv(filename, encoding='utf-8')
        return df

    def col_rename(self):

        self.dict = {'min_idle_s': 'Idle Min',
                     'max_idle_s': 'Idle Max',
                     'std_idle_s': 'Idle Std',
                     'mean_idle_s': 'Idle Mean',
                     'dst_port': 'Destination Port',
                     'duration': 'Duration',
                     'total_fpackets': 'Total Fwd Packets',
                     'total_bpackets': 'Total Backward Packets',
                     'total_fpktl': 'Total Length of Fwd Packets',
                     'min_fpktl': 'Fwd Packet Length Min',
                     'max_fpktl': 'Fwd Packet Length Max',
                     'mean_fpktl': 'Fwd Packet Length Mean',
                     'std_fpktl': 'Fwd Packet Length Std',
                     'min_bpktl': 'Bwd Packet Length Min',
                     'max_bpktl': 'Bwd Packet Length Max',
                     'std_bpktl': 'Bwd Packet Length Std',
                     'mean_bpktl': 'Bwd Packet Length Mean',
                     'flowBytesPerSecond': 'Flow Bytes/s',
                     'flowPktsPerSecond': 'Flow Packets/s',
                     'mean_flowiat': 'Flow IAT Mean',
                     'std_flowiat': 'Flow IAT Std',
                     'max_flowiat': 'Flow IAT Max',
                     'min_flowiat': 'Flow IAT Min',
                     'total_fiat': 'Fwd IAT Total',
                     'mean_fiat': 'Fwd IAT Mean',
                     'std_fiat': 'Fwd IAT Std',
                     'max_fiat': 'Fwd IAT Max',
                     'min_fiat': 'Fwd IAT Min',
                     'total_biat': 'Bwd IAT Total',
                     'max_biat': 'Bwd IAT Max',
                     'fpsh_cnt': 'Fwd PSH Flags',
                     'fPktsPerSecond': 'Fwd Packets/s',
                     'bPktsPerSecond': 'Bwd Packets/s',
                     'min_flowpktl': 'Min Packet Length',
                     'max_flowpktl': 'Max Packet Length',
                     'mean_flowpktl': 'Mean Packet Length',
                     'std_flowpktl': 'Packet Length Std',
                     'var_flowpktl': 'Packet Length Variance',
                     'flow_fin': 'FIN Flag Count',
                     'flow_syn': 'SYN Flag Count',
                     'flow_rst': 'RST Flag Count',
                     'flow_psh': 'PSH Flag Count',
                     'flow_ack': 'ACK Flag Count',
                     'avgPacketSize': 'Average Packet Size',
                     'fAvgSegmentSize': 'Avg Fwd Segment Size',
                     'bAvgSegmentSize': 'Avg Bwd Segment Size',
                     'fSubFlowAvgPkts': 'Subflow Fwd Packets',
                     'fSubFlowAvgBytes': 'Subflow Fwd Bytes',
                     'bSubFlowAvgPkts': 'Subflow Bwd Packets',
                     'bSubFlowAvgBytes': 'Subflow Bwd Bytes',
                     'fInitWinSize': 'Init_Win_bytes_forward',
                     'bInitWinSize': 'Init_Win_bytes_backward',
                     'fDataPkts': 'act_data_pkt_fwd',
                     'fHeaderSizeMin': 'Min Header size_forward',
                     'label': 'Label',
                     'total_fhlen': 'Fwd Header Length',
                     'flow_urg': 'URG Flag Count'
                     }

        # call rename () method
        self.df.rename(columns=self.dict,
                  inplace=True)

        return self.df
    def get_df(self):
        return self.df
    
    def get_df_shape(self):
        return self.df.shape

    def r_hdf(self):
        filename = 'prediction_data/data.h5'
        self.df = pd.read_hdf(filename)
        return self.df

    def save_to_hdf(self):
        # converting df(csv) to df(HDF5)
        filename = 'prediction_data/data.h5'
        self.df.to_hdf(filename, 'data', mode='w', format='table')
        self.df.head(10)
        # return self.df
        
        rc.log("\n[cyan]Converted Df to HDF5[/]\n")
        # del df

    def df_info(self):
        # df.head(5)
        rc.log("\n[purple]Head of Dataframe: {}[/] \n".format(self.df.head(5)))
        # rc.log(self.df.head(5))

        # df.shape
        rc.log("\n[magenta]Shape of Dataframe: {}[/]\n".format(self.df.shape))
        # rc.log(self.df.shape)

        # No. of rows and columns in dataframe
        rc.log(
            "\n[cyan]Number of Rows in Dataframe: {}[/]\n".format(self.df.shape[0]))
        rc.log("\n[good]Number of Columns in Dataframe: {}[/]\n".format(
            self.df.shape[1]))

        # # df.info
        # rc.log("\nDataframe Information: \n")
        # self.df.info()

    def columns_in_df(self):
        rc.log("\n[yellow]Columns in Dataframe:[/] \n")
        col = []
        for i in self.df.columns:
            col.append(i)
        return col

    def dropna(self):
        # df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Dropping all the rows with nan valuess
        self.df.dropna(inplace=True)
        return self.df

    # -------------------changing datatypes

    def check_size_dtypes(self, df):

        max = df.max()
        rc.log('[yellow]Maximum: {}[/]'.format(max))
        # rc.log(max, 'max')

        min = df.min()
        rc.log('[blue]Minimum: {}[/]'.format(min))

        # rc.log(min, 'min')
        # rc.log(df.value_counts())
        var1 = df.memory_usage(index=False, deep=True)
        rc.log('[magenta]This is the memory usage: {}[/]'.format(var1))
        # rc.log(var1, 'This is the memory usage')
        # rc.log(df.sample(8))

    def convert_datatypes(self, df, a='uint8'):

        # rc.log('Trying to convert datatypes for less memory usage')
        max = df.max()
        rc.log('[yellow]Maximum: {}[/]'.format(max))

        min = df.min()
        rc.log('[blue]Minimum: {}[/]'.format(min))

        # rc.log(df.value_counts())

        var1 = df.memory_usage(index=False, deep=True)
        rc.log('[cyan]This is the memory usage: {}[/]'.format(var1))
        df = df.astype(a, errors='ignore')
        var2 = df.memory_usage(index=False, deep=True)
        # rc.log(var2, ' new memory usage| the difference -> ', var1 / var2)
        return df
    
    def check_anomalies(self):
        print(self.df.isnull().values.any())
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Dropping all the rows with nan valuess
        self.df.dropna(inplace=True)
        return self.df.isnull().values.any()
    
    def delColumns(self, col):
        del self.df[col]
        # return self.df
        
    def normAndconv(self,li):
        for i in self.df.columns:
            if i in li:
                print(i)
                # result[i] = dropna(result[i])
                # result[i] = convert_datatypes(result[i])
                self.df[i] = self.df[i].astype('float64', errors='ignore')
            else:
                # print(result[i])
                # result[i] = dropna(result[i])
                # self.df[i] = self.normalize(i)
                self.df[i] = self.convert_datatypes(self.df[i], 'uint8')
                # result[i] = result[i].astype('uint8', errors='ignore')
        return self.df
        
    def colExist(self, col):
        for i in self.df.columns:
            if col in self.df.columns:
                print('exists')
        return self.df

    def normalize(self, i):

        rc.log("[blue][* ] - Normalized data[/]")
        # self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # self.df.dropna(inplace=True)
        normalized_df = ((self.df[i] - self.df[i].min()) /
                         (self.df[i].max() - self.df[i].min())) * 255
        return normalized_df
    
    def get_columns(self):
        return self.df.columns
    
    # def write_log()

    def apply_fn(self):

        self.df['dst_port'] = self.normalize(self.df['dst_port'])
        self.df['dst_port'] = self.convert_datatypes(self.df['dst_port'])
        self.check_size_dtypes(self.df['dst_port'])

        self.df['duration'] = self.normalize(self.df['duration'])
        self.df['duration'] = self.convert_datatypes(self.df['duration'])
        self.check_size_dtypes(self.df['duration'])

        self.df['total_fpackets'] = self.normalize(self.df['total_fpackets'])
        self.df['total_fpackets'] = self.convert_datatypes(self.df['total_fpackets'])
        self.check_size_dtypes(self.df['total_fpackets'])

        self.df['total_bpackets'] = self.normalize(self.df['total_bpackets'])
        self.df['total_bpackets'] = self.convert_datatypes(self.df['total_bpackets'])
        self.check_size_dtypes(self.df['total_bpackets'])

        self.df['total_fpktl'] = self.normalize(self.df['total_fpktl'])
        self.df['total_fpktl'] = self.convert_datatypes(self.df['total_fpktl'])
        self.check_size_dtypes(self.df['total_fpktl'])

        self.df['min_fpktl'] = self.normalize(self.df['min_fpktl'])
        self.df['min_fpktl'] = self.convert_datatypes(self.df['min_fpktl'])
        self.check_size_dtypes(self.df['min_fpktl'])

        self.df['max_fpktl'] = self.normalize(self.df['max_fpktl'])
        self.df['max_fpktl'] = self.convert_datatypes(self.df['max_fpktl'])
        self.check_size_dtypes(self.df['max_fpktl'])

        self.df['mean_fpktl'] = self.normalize(self.df['mean_fpktl'])
        self.df['mean_fpktl'] = self.convert_datatypes(self.df['mean_fpktl'])
        self.check_size_dtypes(self.df['mean_fpktl'])

        self.df['std_fpktl'] = self.normalize(self.df['std_fpktl'])
        self.df['std_fpktl'] = self.convert_datatypes(self.df['std_fpktl'])
        self.check_size_dtypes(self.df['std_fpktl'])

        self.df['min_bpktl'] = self.normalize(self.df['min_bpktl'])
        self.df['min_bpktl'] = self.convert_datatypes(self.df['min_bpktl'])
        self.check_size_dtypes(self.df['min_bpktl'])

        self.df['max_bpktl'] = self.normalize(self.df['max_bpktl'])
        self.df['max_bpktl'] = self.convert_datatypes(self.df['max_bpktl'])
        self.check_size_dtypes(self.df['max_bpktl'])

        self.df['std_bpktl'] = self.normalize(self.df['std_bpktl'])
        self.df['std_bpktl'] = self.convert_datatypes(self.df['std_bpktl'])
        self.check_size_dtypes(self.df['std_bpktl'])

        self.df['flowBytesPerSecond'] = self.normalize(self.df['flowBytesPerSecond'])
        self.df['flowBytesPerSecond'] = self.convert_datatypes(
            self.df['flowBytesPerSecond'])
        self.check_size_dtypes(df['flowBytesPerSecond'])

        # self.df['flowPktsPerSecond'] = d.normalize(self.df['flowPktsPerSecond'])
        # self.df['flowPktsPerSecond'] = d.convert_datatypes(self.
        #     self.df['flowPktsPerSecond'])
        # d.check_size_dtypes(self.df['flowPktsPerSecond'])

        self.df['mean_flowiat'] = self.normalize(self.df['mean_flowiat'])
        self.df['mean_flowiat'] = self.convert_datatypes(self.df['mean_flowiat'])
        self.check_size_dtypes(self.df['mean_flowiat'])

        self.df['std_flowiat'] = self.normalize(self.df['std_flowiat'])
        self.df['std_flowiat'] = self.convert_datatypes(self.df['std_flowiat'])
        self.check_size_dtypes(self.df['std_flowiat'])

        self.df['max_flowiat'] = self.normalize(self.df['max_flowiat'])
        self.df['max_flowiat'] = self.convert_datatypes(self.df['max_flowiat'])
        self.check_size_dtypes(self.df['max_flowiat'])

        self.df['min_flowiat'] = self.normalize(self.df['min_flowiat'])
        self.df['min_flowiat'] = self.convert_datatypes(self.df['min_flowiat'])
        self.check_size_dtypes(self.df['min_flowiat'])

        self.df['total_fiat'] = self.normalize(self.df['total_fiat'])
        self.df['total_fiat'] = self.convert_datatypes(self.df['total_fiat'])
        self.check_size_dtypes(self.df['total_fiat'])

        self.df['mean_fiat'] = self.normalize(self.df['mean_fiat'])
        self.df['mean_fiat'] = self.convert_datatypes(self.df['mean_fiat'])
        self.check_size_dtypes(self.df['mean_fiat'])

        self.df['std_fiat'] = self.normalize(self.df['std_fiat'])
        self.df['std_fiat'] = self.convert_datatypes(self.df['std_fiat'])
        self.check_size_dtypes(self.df['std_fiat'])

        self.df['max_fiat'] = self.normalize(self.df['max_fiat'])
        self.df['max_fiat'] = self.convert_datatypes(self.df['max_fiat'])
        self.check_size_dtypes(self.df['max_fiat'])

        self.df['min_fiat'] = self.normalize(self.df['min_fiat'])
        self.df['min_fiat'] = self.convert_datatypes(self.df['min_fiat'])
        self.check_size_dtypes(self.df['min_fiat'])

        self.df['total_biat'] = self.normalize(self.df['total_biat'])
        self.df['total_biat'] = self.convert_datatypes(self.df['total_biat'])
        self.check_size_dtypes(df['total_biat'])

        self.df['max_biat'] = self.normalize(self.df['max_biat'])
        self.df['max_biat'] = self.convert_datatypes(self.df['max_biat'])
        self.check_size_dtypes(self.df['max_biat'])

        self.df['fpsh_cnt'] = self.normalize(self.df['fpsh_cnt'])
        self.df['fpsh_cnt'] = self.convert_datatypes(self.df['fpsh_cnt'])
        self.check_size_dtypes(self.df['fpsh_cnt'])

        self.df['fPktsPerSecond'] = self.normalize(self.df['fPktsPerSecond'])
        self.df['fPktsPerSecond'] = self.convert_datatypes(self.df['fPktsPerSecond'])
        self.check_size_dtypes(self.df['fPktsPerSecond'])

        self.df['bPktsPerSecond'] = self.normalize(self.df['bPktsPerSecond'])
        self.df['bPktsPerSecond'] = self.convert_datatypes(self.df['bPktsPerSecond'])
        self.check_size_dtypes(self.df['bPktsPerSecond'])

        self.df['min_flowpktl'] = self.normalize(self.df['min_flowpktl'])
        self.df['min_flowpktl'] = self.convert_datatypes(self.df['min_flowpktl'])
        self.check_size_dtypes(self.df['min_flowpktl'])

        self.df['max_flowpktl'] = self.normalize(self.df['max_flowpktl'])
        self.df['max_flowpktl'] = self.convert_datatypes(self.df['max_flowpktl'])
        self.check_size_dtypes(self.df['max_flowpktl'])

        self.df['mean_flowpktl'] = self.normalize(self.df['mean_flowpktl'])
        self.df['mean_flowpktl'] = self.convert_datatypes(self.df['mean_flowpktl'])
        self.check_size_dtypes(self.df['mean_flowpktl'])

        self.df['std_flowpktl'] = self.normalize(self.df['std_flowpktl'])
        self.df['std_flowpktl'] = self.convert_datatypes(self.df['std_flowpktl'])
        self.check_size_dtypes(self.df['std_flowpktl'])

        self.df['var_flowpktl'] = self.normalize(self.df['var_flowpktl'])
        self.df['var_flowpktl'] = self.convert_datatypes(self.df['var_flowpktl'])
        self.check_size_dtypes(self.df['var_flowpktl'])

        self.df['flow_fin'] = self.normalize(self.df['flow_fin'])
        self.df['flow_fin'] = self.convert_datatypes(self.df['flow_fin'])
        self.check_size_dtypes(self.df['flow_fin'])

        self.df['flow_syn'] = self.normalize(self.df['flow_syn'])
        self.df['flow_syn'] = self.convert_datatypes(self.df['flow_syn'])
        self.check_size_dtypes(self.df['flow_syn'])

        self.df['flow_rst'] = self.normalize(self.df['flow_rst'])
        self.df['flow_rst'] = self.convert_datatypes(self.df['flow_rst'])
        self.check_size_dtypes(self.df['flow_rst'])

        self.df['flow_psh'] = self.normalize(self.df['flow_psh'])
        self.df['flow_psh'] = self.convert_datatypes(self.df['flow_psh'])
        self.check_size_dtypes(self.df['flow_psh'])

        self.df['flow_ack'] = self.normalize(self.df['flow_ack'])
        self.df['flow_ack'] = self.convert_datatypes(self.df['flow_ack'])
        self.check_size_dtypes(df['flow_ack'])

        self.df['avgPacketSize'] = self.normalize(self.df['avgPacketSize'])
        self.df['avgPacketSize'] = self.convert_datatypes(self.df['avgPacketSize'])
        self.check_size_dtypes(self.df['avgPacketSize'])

        self.df['fAvgSegmentSize'] = self.normalize(self.df['fAvgSegmentSize'])
        self.df['fAvgSegmentSize'] = self.convert_datatypes(self.df['fAvgSegmentSize'])
        self.check_size_dtypes(self.df['fAvgSegmentSize'])

        self.df['bAvgSegmentSize'] = self.normalize(self.df['bAvgSegmentSize'])
        self.df['bAvgSegmentSize'] = self.convert_datatypes(self.df['bAvgSegmentSize'])
        self.check_size_dtypes(self.df['bAvgSegmentSize'])

        self.df['fSubFlowAvgPkts'] = self.normalize(self.df['fSubFlowAvgPkts'])
        self.df['fSubFlowAvgPkts'] = self.convert_datatypes(self.df['fSubFlowAvgPkts'])
        self.check_size_dtypes(self.df['fSubFlowAvgPkts'])

        self.df['fSubFlowAvgBytes'] = self.normalize(self.df['fSubFlowAvgBytes'])
        self.df['fSubFlowAvgBytes'] = self.convert_datatypes(self.df['fSubFlowAvgBytes'])
        self.check_size_dtypes(self.df['fSubFlowAvgBytes'])

        self.df['bSubFlowAvgPkts'] = self.normalize(self.df['bSubFlowAvgPkts'])
        self.df['bSubFlowAvgPkts'] = self.convert_datatypes(self.df['bSubFlowAvgPkts'])
        self.check_size_dtypes(self.df['bSubFlowAvgPkts'])

        self.df['bSubFlowAvgBytes'] = self.normalize(self.df['bSubFlowAvgBytes'])
        self.df['bSubFlowAvgBytes'] = self.convert_datatypes(self.df['bSubFlowAvgBytes'])
        self.check_size_dtypes(self.df['bSubFlowAvgBytes'])

        self.df['fInitWinSize'] = self.normalize(self.df['fInitWinSize'])
        self.df['fInitWinSize'] = self.convert_datatypes(self.df['fInitWinSize'])
        self.check_size_dtypes(self.df['fInitWinSize'])

        self.df['bInitWinSize'] = self.normalize(self.df['bInitWinSize'])
        self.df['bInitWinSize'] = self.convert_datatypes(self.df['bInitWinSize'])
        self.check_size_dtypes(self.df['bInitWinSize'])

        self.df['fDataPkts'] = self.normalize(self.df['fDataPkts'])
        self.df['fDataPkts'] = self.convert_datatypes(self.df['fDataPkts'])
        self.check_size_dtypes(self.df['fDataPkts'])

        df['fHeaderSizeMin'] = self.normalize(self.df['fHeaderSizeMin'])
        df['fHeaderSizeMin'] = self.convert_datatypes(self.df['fHeaderSizeMin'])
        self.check_size_dtypes(self.df['fHeaderSizeMin'])

        self.df['total_fhlen'] = self.normalize(self.df['total_fhlen'])
        self.df['total_fhlen'] = self.convert_datatypes(self.df['total_fhlen'])
        self.check_size_dtypes(self.df['total_fhlen'])

        self.df['min_idle_s'] = self.normalize(self.df['min_idle_s'])
        self.df['min_idle_s'] = self.convert_datatypes(self.df['min_idle_s'])
        self.check_size_dtypes(self.df['min_idle_s'])

        self.df['max_idle_s'] = self.normalize(self.df['max_idle_s'])
        self.df['max_idle_s'] = self.convert_datatypes(self.df['max_idle_s'])
        self.check_size_dtypes(self.df['max_idle_s'])

        self.df['std_idle_s'] = self.normalize(self.df['std_idle_s'])
        self.df['std_idle_s'] = self.convert_datatypes(self.df['std_idle_s'])
        self.check_size_dtypes(self.df['std_idle_s'])

        self.df['mean_idle_s'] = self.normalize(self.df['mean_idle_s'])
        self.df['mean_idle_s'] = self.convert_datatypes(self.df['mean_idle_s'])
        self.check_size_dtypes(self.df['mean_idle_s'])

        self.df['flow_urg'] = self.normalize(self.df['flow_urg'])
        self.df['flow_urg'] = self.convert_datatypes(self.df['flow_urg'])
        self.check_size_dtypes(self.df['flow_urg'])

        # self.df['fHeaderSizeMin'] = self.d.normalize(self.df['fHeaderSizeMin'])
        # self.df['fHeaderSizeMin'] = self.d.convert_datatypes(self.df['fHeaderSizeMin'])
        # d.check_size_dtypes(self.df['fHeaderSizeMin'])

        '''
            39  URG Flag Count               87 non-null     float64 'flow_urg'
            26  Fwd Header Length            0 non-null      float64  'fHeaderSizeMin'

                Idle Mean                    87 non-null     float64  'mean_idle_s', 'std_idle_s', 'max_idle_s', 'min_idle_s'
            52  Idle Std                     87 non-null     float64
            53  Idle Max                     87 non-null     float64
            54  Idle Min                     87 non-null     float64

        '''

        rc.log("[magenta]Data information: \n{}[/]".format(df.info()))


if __name__ == "__main__":

    filename = 'csvs/merged_data.csv'

    df = pd.read_csv(filename)
    # print(df.head(100))
    d = Preprocessing(df)
    d.rm_col()
    d.col_rename()
    d.dropna()
    # d.rm_col()
    
    li = [' Destination Port',
        'Flow Bytes/s','Flow Packets/','Fwd Packets/s','Bwd Packets/s',
        'Fwd IAT Std','Bwd IAT Std','Flow IAT Std','Packet Length Variance',
        'Bwd IAT Max','Bwd Packets/s','Active Mean',  'Flow Duration',
        'Total Fwd Packets', 'Active Std','Active Max','Active Min','Idle Mean',
        'Idle Std','Idle Min','Idle Max','Fwd Packets/s']
    
    
    

    # d.delColumns(df['SYN Flag Count'])
    # d.delColumns(df['Init_Win_bytes_backward'])
    
    # print(g.head(100))
    # print('heeeeeeeeeeeeeeeeeeey')
    rc.log("[blue]Shape of captured data: \n{}[/]".format(d.get_df_shape()))

    # d.check_anomalies()
    # g = d.get_df()
    # print(g.head(100))
    rc.log("[cyan][*_*] - Preprocessing the captured Data - [*_*][/]\n\n")

    d.check_anomalies()
    # d.normAndconv(li)
    d.save_to_hdf()
    
    # df['label']
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Dropping all the rows with nan valuess
    # df.dropna(inplace=True)
    

    rc.log("[yellow]Columns/ Features in captured data: \n{}[/]".format(d.get_columns()))
    rc.log("[blue]Shape of captured data: \n{}[/]".format(df.shape))
    print(len(d.get_columns()),d.get_columns(), 'cplll')
    # df.columns
    # del df['SYN Flag Count']
    # del df['Init_Win_bytes_backward']
    
    
    # ---------------------------
    
    # d.df_info()
    
    # d.dropna()
    # d.rm_col()
    # d.columns_in_df()
    # 
    # l = d.col_rename()

    # col = []
    # for i in l:
        # col.append(i)
        # rc.log("[cyan]<------- {} ------->[/]".format(i))
    # l.info()

    # rc.log("[good][ DONE ] - File is ready to fed to ML/ DL model. [magenta][ *in HDF5 Format ][/][/]")
# 
    # a = len(df.columns)
    # print(df.info(), a, 'df infoooo')
    # 
    # 
# 
    # 
    # rc.log(df.columns)
    # rc.log("[bold blue][ *** ] - Shape of captured data: {}[/]".format(df.shape))

    # rc.save_html("norm-report.html")
    # 
    # print(df.columns)
    # print(df.isnull().values.any())
    # d.check_anomalies()
    
    # ll = d.save_to_hdf()
    # print(ll.head(10))
    # ==============================================
    
    
    # df = pd.read_csv(filename)
    # rc.log("[cyan][*_*] - Preprocessing the captured Data - [*_*][/]\n\n")

    # rc.log("[yellow]Columns/ Features in captured data: \n{}[/]".format(df.columns))
    # # df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    # rc.log("[blue]Shape of captured data: \n{}[/]".format(df.shape))
    # d = Preprocessing(df)
    # df = d.r_csv(filename)

    # rc.log(df)

    # rc.log("[purple]Droping NaN values.....\n[/]")
    # d.dropna()

    # # # df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    # # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # # # Dropping all the rows with nan valuess
    # # df.dropna(inplace=True)

    # rc.log("[bold bad]Removing unwanted columns...\n[/]")
    # d.rm_col()

    # d.apply_fn()

    # rc.log("[good]Giving Columns a meaningful name...[/]")
    # l = d.col_rename()

    # rc.log("[cyan]Data info(): \n[/] \n")
    # l.info()

    # rc.log("[magenta][*_*] - Saving preprocessed_data.csv,..[/]")
    # df.to_csv("preprocessed_csv/preprocessed_data.csv", encoding='utf-8')
    # rc.log("\n[good][*_*] - Saved Preprocessed csv[/]\n")

    # rc.log("[good][ DONE ] - File is ready to fed to ML/ DL model. [magenta][ *in HDF5 Format ][/][/]")
    # d.save_to_hdf()
    # rc.log(df.columns)
    # rc.log("[bold blue][ *** ] - Shape of captured data: {}[/]".format(df.shape))

    # rc.save_html("norm-report.html")
