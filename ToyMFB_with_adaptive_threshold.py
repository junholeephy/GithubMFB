import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import scipy
import statsmodels
from collections import deque, Counter
from scipy.signal import savgol_filter
from scipy.stats.mstats import theilslopes
from collections import deque
from matplotlib.lines import Line2D
import time

from BOCD import bocd_meanNstd, NormalUnKnownMeanPrecision, Batch_SlopeT_SteadyState
from BOCD import bocd, NormalUnKnownMeanPrecision


def HotellingT2(window):
    alpha = 0.01
    p = 1
    m = len(window)
    q = 2 * (m - 1) ** 2 / (3 * m - 4)

    UCL = (m - 1) ** 2 * stats.beta.isf(alpha, p / 2, (q - p - 1) / 2) / m

    mean = np.mean(window)
    V = np.array([])
    T2_list = []

    for ind in range(m - 1):
        V = np.append(V, window[ind + 1] - window[ind])

    S = np.array([0.5 * V.transpose() @ V / (m - 1)])

    for item in window:
        delta = np.array(item) - np.array(mean)
        T2 = delta * np.linalg.inv(np.array([S])) * delta
        T2_list.append(T2)

    anomaly = []
    for ind, value in enumerate(T2_list):
        if value > UCL:
            anomaly.append(ind)

    return anomaly


class ToyMFB(object):
    def __init__(self, Nw=15, No=20, Nstd=30):
        '''
        :param Nw: window size for outlier
        :param No: window for waiting Mounter-FeedBack (MFB) reflection
        :param Nstd: window for updating std.
        '''

        # status : two status considered i.e. 'Default' & 'DidOperation'
        self.status = 'Default'

        # window size
        self.Nw = Nw  # mentioned
        self.No = No  # mentioned
        self.Ncpd = self.No + self.Nw  # window size for detecting change-point
        self.Nstd = Nstd  # mentioned

        # data list
        self.total_data = []  # total data list after feedback
        self.op_list = []  # indexes where the operation was applied
        self.cp_list = []  # indexes of real change-points
        self.CPD_list = [0]  # indexes of detected change-points
        self.sub_op_list = []  # indexes where the operation was applied

        # data window
        self.window = deque([], maxlen=self.Nw)  # Raw data list for calculating offset & for smoothing
        self.window_for_cpd = deque([], maxlen=self.Ncpd)  # Raw data list for detecting change-point
        self.window_for_compare = deque([], maxlen=self.Nw)  # Smoothed data list to compare target & current state

        # Target & Threshold
        self.Target = 0
        self.offset_threshold = None

        # Operation
        self.operation = 0  # cumulative sum of feedback operation
        self.operation_buffer = 0  # buffer of feedback operation to wait for delay
        self.operation_cnt = 0  # number of operation
        self.sub_operation_cnt = 0  # number of sub operation

        # Operation Delay
        self.mean_D = 5
        self.D = None
        self.delay = 0  # number of steps that have elapsed since the operation.

        # waiting step for reflection
        self.wait_num = 1  # number counting for monitoring Mounter-FeedBack delay

    def _run_MeanCPD_f(self):
        '''
        Check if change has occurred in the latest (No + Nw) data
        :return: True/False, sequence point of CP
        '''
        # BOCD 가동
        TimeSequence = self.Ncpd  # window size for detecting change-point (mentioned)
        std = np.std(np.array(self.window_for_cpd)[:self.Nw])  # Std calculation to be input for BOCPD

        mu0 = 0  # Prior on Gaussian mean. (A parameter of normal-gamma prior distribution)
        gamma0 = 1  # (A parameter of normal-gamma prior distribution)
        alpha0 = 1  # (A parameter of normal-gamma prior distribution)
        beta0 = alpha0 * std ** 2  # "sqrt(beta/alpha) = std" (A parameter of normal-gamma prior distribution)

        hazard = 1 / 50.0  # Hazard survival function assumes probability to be a CP for each data point.
        message = np.array([1])  # Iterative message calculated using previously collected data.

        Data_list = []
        RL_dist_list = []
        Temp_RL_dist = np.zeros((TimeSequence, TimeSequence))

        model = NormalUnKnownMeanPrecision(mu0, gamma0, alpha0, beta0)  # Data assumed be to a normal distribution.
        # Case : Both of mean and std unknown
        # BOCPD class called.

        RL = []  # Run-Length distribution initial list
        Start = []  # Estimated Change-Point starting point
        cp_list = [0]  # Detected Change-Point sequence saved list

        for ind, cont in enumerate(range(TimeSequence)):
            Data_list.append(self.window_for_cpd[ind])
            RL_dist, new_joint = bocd_meanNstd(data_list=Data_list, model=model,
                                               hazard=hazard, Message=message)

            message = new_joint  # each point sequential likelihood, or numerator of RL posterior distribution.
            RL_dist_list.append(RL_dist)
            Temp_RL_dist[ind, :ind + 1] = RL_dist[1:]  # RL distribution temporary saved as a element of a list

            RL.append(np.argmax(
                Temp_RL_dist[ind]))  # The highest probability corresponding sequence of RL posterior distribution
            Start.append((ind, ind - RL[-1]))

            if ind >= 1:
                if Start[-1][1] != Start[-2][1]:
                    if max(cp_list) > ind - RL[-1]:
                        cp_list = cp_list[:-1]  # Remove the lastly added Change-Point
                    else:
                        cp_list.append(ind - RL[-1])  # Change-point Detected and append into the list

        if not cp_list:  # if no FeedBack operation application found
            return False, None
        elif cp_list[-1] < self.Nw:  # Exclude a CP ahead of FeedBack operation applied sequence
            return False, None
        else:
            for cp in cp_list[1:]:  # Exclude initial point (the point '0')
                self.CPD_list.append(len(self.total_data) - self.Ncpd + cp_list[-1])
            return True, cp_list[-1]

    def _is_on_target(self):
        '''
        Check if the smoothed data is inside the threshold
        :return: True/ False
        '''
        is_target = False

        # Outlier : Last Nw smoothed data must be the same sign
        check = 0
        for smoothed in self.window_for_compare:
            check += np.sign(smoothed)
        if np.abs(check) != self.Nw:
            is_target = True

        # Outlier : Last Nw smoothed data must be out of threshold.
        for smoothed in self.window_for_compare:
            if np.abs(smoothed - self.Target) <= self.offset_threshold:
                is_target = True

        return is_target

    def _do_operation(self):
        '''
        Apply operation & Set operation delay
        OUTPUT : None
        '''
        self.operation_cnt += 1  # number of operations
        self.sub_operation_cnt = 0

        self.D = round(np.random.exponential(scale=self.mean_D))  # random delay : exp.dist. with mean 5
        self.D = 0
        self.delay = 0  # reset self.delay

        # exclude anomaly before calculating offset by using Hotelling's T2
        buffer = self.window
        anomaly = HotellingT2(buffer)
        for cnt, ind in enumerate(anomaly):
            del buffer[ind - cnt]

        # calculate offset & operation
        offset = np.mean(buffer)
        self.operation_buffer = self.Target - offset

        # append operation index
        self.op_list.append(len(self.total_data))

    def _do_sub_operation(self):
        '''
        Apply sub-operation & Set sub-operation delay
        OUTPUT : None
        '''
        #         self.operation_cnt += 1  # number of operations
        self.sub_operation_cnt += 1  # number of sub operations

        self.D = round(np.random.exponential(scale=self.mean_D))  # random delay : exp.dist. with mean 5
        self.D = 0
        self.delay = 0  # reset self.delay

        # exclude anomaly before calculating offset by using Hotelling's T2
        buffer = self.total_data[self.CPD_list[-1]:]
        anomaly = HotellingT2(buffer)
        for cnt, ind in enumerate(anomaly):
            del buffer[ind - cnt]

        # calculate offset & operation
        offset = np.mean(buffer)
        self.operation_buffer = self.Target - offset

        # append operation index
        self.sub_op_list.append(len(self.total_data))

    def step(self, feature):
        '''
        MFB Loop
        :param feature: offset data
        :return: None
        '''
        # START
        if self.status == 'Default':
            self._put_feature(feature)

            # Enough data ?
            if len(self.total_data) < self.Nw:
                return

            # Is on target? If not, do operation.
            if self._is_on_target():
                if self.sub_operation_cnt == 0 and (len(self.total_data) - self.CPD_list[-1]) >= 15:
                    if adfuller(self.total_data[self.CPD_list[-1]:])[1] < 1e-6:
                        self._do_sub_operation()
                return
            else:
                self._do_operation()
                self.status = 'DidOperation'
                return

        # Waiting for results to be reflected.
        if self.status == 'DidOperation':
            if self.wait_num < self.No:
                self._put_feature(feature)
                self.wait_num += 1
                return
            else:
                self._put_feature(feature)
                self.wait_num = 1

            # Mean CPD-f
            self._run_MeanCPD_f()

            self.status = 'Default'
            return

    def show_param(self):
        '''
        Show parameters
        :return: None
        '''
        print('>> Parameters :')
        print('\tNw =', self.Nw, ', No =', self.No, '\n\tNcpd =', self.Ncpd, '(= Nw+No)')
        print('\tMean Delay =', self.mean_D)
        print('\tTarget Mean =', self.Target)
        print('\tOffset Threshold =', self.offset_threshold)
        print('\n=============================================')

    def draw(self):
        '''
        Draw figures
        :return: None
        '''
        # Plot total data & operation index & cp index
        fit, (ax1) = plt.subplots(nrows=2, ncols=1, figsize=(20, 8))
        ax1[0].grid(True)
        for ind in self.op_list:
            ax1[0].axvline(ind, linestyle='--', color='g')
        for ind in self.sub_op_list:
            ax1[0].axvline(ind, linestyle=':', color='r')
        for ind in self.cp_list:
            ax1[0].axvline(ind, linestyle=':', color='b')
        ax1[0].plot(self.total_data)
        ax1[0].axhline(0, linestyle='--', color='grey')
        legend_elements = [Line2D([0], [0], color='green', lw=2, label='Operation'),
                           Line2D([0], [0], color='blue', lw=2, label='After Delay')]
        plt.legend(handles=legend_elements)

        # Plot total data & detected cp index
        ax1[1].plot(self.total_data)
        ax1[1].grid(True)
        ax1[1].axhline(0, linestyle='--', color='grey')
        for ind in self.op_list:
            ax1[1].axvline(ind - self.Nw, linestyle='--', color='black')
        for ind in self.op_list:
            if ind + self.No < len(self.total_data):
                ax1[1].axvline(ind + self.No, linestyle='--', color='grey')
        for ind in self.CPD_list[1:]:
            ax1[1].axvline(ind, linestyle='--', color='r', linewidth=1)
        plt.title('CPD-f result')
        legend_elements = [Line2D([0], [0], color='red', lw=2, label='CP After Op.')]
        plt.legend(handles=legend_elements)

    def _put_feature(self, feature):
        '''
        Get offset data
        :param feature: offset data
        :return: None
        '''

        # Apply operation after delay D
        if self.delay == self.D:
            self.operation += self.operation_buffer
        #             self.cp_list.append(len(self.total_data))

        feature += self.operation
        self.delay += 1

        # Append feature to data lists
        self.total_data.append(feature)
        self.window.append(feature)
        self.window_for_cpd.append(feature)

        # Initialize the threshold
        if len(self.total_data) == self.Nw:
            self.offset_threshold = 0.5 * np.std(self.total_data)
            print(len(self.total_data), '\t', (self.offset_threshold))

        # Data smoothing
        if len(self.total_data) >= self.Nw:
            savgol_result = savgol_filter(self.total_data[-self.Nw:], self.Nw, 3)
            self.window_for_compare = savgol_result[-self.Nw:]

        # Reset the threshold
        if len(self.cp_list) != 0:
            if (len(self.total_data) - self.cp_list[-1]) % self.Nstd == 0:
                buffer = self.total_data[-self.Nstd:]
                #                 slope = theilslopes(buffer, alpha = 0.99)[0]
                slope = 0

                buffer = [buffer[i] - slope * i for i in range(self.Nstd)]

                if 0.7 * self.offset_threshold > np.std(buffer) or self.offset_threshold < 0.7 * np.std(buffer):
                    self.offset_threshold = 0.5 * np.std(buffer)


if __name__ == '__main__':
    # Set test data
    Data_temp = pd.read_csv("C:/Users/T5402/Downloads/n1_test_500k.txt", sep=',')
    print(Data_temp.columns, '\n\n')

    Pad_Temp = Data_temp[(Data_temp.Array_index == 1) & (Data_temp.Component_Name == 'SW3')]
    PadOff_X = Pad_Temp.PAD_Length_offset / 1000.0
    PadOff_Y = Pad_Temp.PAD_Width_offset / 1000.0
    PadOff_A = Pad_Temp.PAD_Angle_offset / 1000.0

    test_data = np.array(PadOff_X[:])
    test_loop = ToyMFB(Nw=15, No=20)
    test_loop.show_param()

    #     test_data = [test_data[i]+i for i in range(300)]+[test_data[i+300]-2*i+300 for i in range(200)]
    test_data = [test_data[i] + 1 * i for i in range(200)]

    start = time.time()  # set start time

    # Run feedback loop
    for item in test_data:
        test_loop.step(item)

    print('\n>> time : ', time.time() - start, '(s)')  # print working time

    # Show RMS score before & after feedback
    RMS_before_FB = np.sqrt(sum((np.array(test_data) - np.zeros(len(test_data))) ** 2) / len(test_data))
    RMS_after_FB = np.sqrt(sum((np.array(test_loop.total_data) - np.zeros(len(test_data))) ** 2) / len(test_data))

    print('\n')
    print('operation count\t: ', test_loop.operation_cnt)
    print('RMS_before_FB\t: ', RMS_before_FB)
    print('RMS_after_FB\t: ', RMS_after_FB)

    # Plot raw data & feedback result
    fit, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    ax1.grid(True)
    plt.title('RMS before FB : ' + str(RMS_before_FB) + '\n RMS after FB : ' + str(RMS_after_FB))
    ax1.plot(test_data[:], 'grey')
    ax1.plot(test_loop.total_data[:], 'b')
    ax1.axhline(0, linestyle='--', color='grey')
    ax1.legend(['Raw Data', 'After Feedback'])

    test_loop.draw()
