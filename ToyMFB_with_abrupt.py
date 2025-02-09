import numpy as np
from collections import deque
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from scipy.signal import savgol_filter
from math import floor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from BOCD import bocd_meanNstd, NormalUnKnownMeanPrecision


def HotellingT2(window):
    '''
    Find indexes of anomaly using Hotelling T-square method
    calculate T2 score for each point & compare with calculated Upper-confidence-level.
    if exceed, anomaly

    :param window: data list
    :return: indexes of anomaly
    '''
    alpha = 0.01  #FIXME : must be larger value
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
    def __init__(self, ARRAY_INDEX='UNKNOWN', REFERENCE='UNKNOWN',AXIS='UNKNOWN', Nw=15, Nomin=3, Nomax=20, Nstd=15):
        '''
        :param ARRAY_INDEX: array index of component
        :param REFERENCE: reference of component
        :param AXIS: padoverhang axis
        :param Nw: window size for outlier
        :param Nomin: min window size for waiting Mounter-FeedBack (MFB) reflection
        :param Nomax: max window size for waiting Mounter-FeedBack (MFB) reflection
        :param Nstd: window size for updating std.
        '''
    
        # Monitored Component (only used in plotting.)
        self.ARRAY_INDEX = ARRAY_INDEX
        self.REFERENCE = REFERENCE
        self.AXIS = AXIS
        
        # status : two status considered i.e. 'noFBapplied' & 'FBapplied'
        self.status = 'noFBapplied'

        # window size
        self.Nw = Nw  # mentioned
        self.Nomin = Nomin  # mentioned
        self.Nomax = Nomax  # mentioned
        self.Nstd = Nstd  # mentioned

        # data list
        self.total_data = []  # total data list (FB applied list : collected data list if online production)
        self.op_list = [0]  # indexes where operation were applied (not considered delay operation list)
        self.cp_list = [0]  # indexes of machine applied real change-points
        self.CPD_list = [0]  # indexes of detected change-points
        self.bocd_f_ind_list = [0]  # indexes of CPD fired points
        self.sub_op_list = [0]  # indexes where the sub-operation was applied

        # data window
        self.window = deque([], maxlen=self.Nw)  # Raw data list for calculating offset & for smoothing (Nw data)
        self.window_for_compare = deque([], maxlen=self.Nw)  # Smoothed data list to compare threshold & current state

        # Target & Threshold
        self.Target = 0
        self.offset_threshold = None

        # Operation
        self.operation = 0  # cumulative sum of feedback operation
        self.operation_buffer = 0  # buffer of feedback operation to wait for delay
        self.operation_cnt = 0  # number of operation
        self.sub_operation = False  # where or not already applied sub-operation, as sub-op applied only once right after CPD

        # Operation Delay
        self.mean_D = 5  # mean operation delay
        self.D = None  # operation delay
        self.delay = 0  # number of steps that have elapsed since the operation.

        # waiting step for reflection
        self.wait_num = 0  # counting number for monitoring Mounter-FeedBack delay
        self.abrupt_num = 0  # number counting low p-value point for Abrupt CPD
        self.p = 0.05  # respective lower & upper p-value for Abrupt CPD
        self.estimator = None  # Gaussian KDE for Abrupt CPD => p-value calculated based on the Gaussian KDE regressed distribution

    def _run_CPD(self):
        '''
        Check if change has occurred in the latest (No + Nw) data
        :return: True/False, sequence point of CP
        '''
        # BOCD 가동
        TimeSequence = self.Nw + len(self.total_data) - self.op_list[-1]  # window size for detecting change-point

        std = np.std(np.array(self.total_data[-TimeSequence:][:self.Nw]))  # Std calculation for BOCPD FIXME (:self.Nw)

        mu0 = 0  # Prior on Gaussian mean. (A parameter of normal-gamma prior distribution)
        gamma0 = 1  # (A parameter of normal-gamma prior distribution)
        alpha0 = 1  # (A parameter of normal-gamma prior distribution)
        beta0 = alpha0 * std ** 2  # "sqrt(beta/alpha) = std" (A parameter of normal-gamma prior distribution)

        hazard = 1 / 50.0  # Hazard survival function assumes probability to be a CP for each data point. #FIXME (1/20.0 ??)
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
            Data_list.append(self.total_data[-TimeSequence + ind])
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

        if not cp_list:  # If no FeedBack candidate found
            return False, None
        elif cp_list[-1] < self.Nw:  # Exclude a CP ahead of FeedBack operation applied sequence
            return False, None
        else:
            for cp in cp_list[1:]:  # Exclude initial point (the point '0')
                self.CPD_list.append(len(self.total_data) - TimeSequence + cp_list[-1])
            return True, cp_list[-1]

    def _is_within_threshold(self):
        '''
        Check if the smoothed data is inside the threshold
        :return: True/ False ('True' if data lies within threshold )
        '''
        is_withinThreshold = False

        # Outlier : Last Nw smoothed data must be the same sign
        '''
        check = 0
        for smoothed in self.window_for_compare:
            check += np.sign(smoothed)

        if np.abs(check) != self.Nw:
            is_withinThreshold = True  
        '''
        check = np.sign(self.window_for_compare[0])
        for smoothed in self.window_for_compare:
            if(check != np.sign(smoothed)):
                is_withinThreshold = True; break; #FIXME (Non-same-sign which is anomaly ??)

        # Outlier : Last Nw smoothed data must be out of threshold.
        for smoothed in self.window_for_compare:
            if np.abs(smoothed - self.Target) <= self.offset_threshold:
                is_withinThreshold = True

        return is_withinThreshold

    def _do_operation(self):
        '''
        Apply operation & Set operation delay
        OUTPUT : None
        '''
        self.operation_cnt += 1  # number of operations
        self.sub_operation = False

        self.D = round(np.random.exponential(scale=self.mean_D))  # random delay : exp.dist. with mean 5
        self.delay = 0  # Reset self.delay

        # Exclude anomaly before calculating offset by using Hotelling's T2
        windowForAnomaly = self.window
        anomaly = HotellingT2(windowForAnomaly)
        for delete_ind, anomaly_ind in enumerate(anomaly):
            del windowForAnomaly[anomaly_ind - delete_ind] # windowForAnomaly : Anomaly excluded window 

        # Calculate offset & operation
        offset = np.mean(windowForAnomaly)
        self.operation_buffer = self.Target - offset

        # Append operation index
        self.op_list.append(len(self.total_data))

    def _do_sub_operation(self):
        '''
        Apply sub-operation & Set sub-operation delay
        OUTPUT : None
        '''
        self.sub_operation = True  # number of sub operations

        self.D = round(np.random.exponential(scale=self.mean_D))  # random delay : exp.dist. with mean 5
        self.delay = 0  # Reset self.delay

        # Exclude anomaly before calculating offset by using Hotelling's T2
        windowForAnomaly = self.total_data[max(self.CPD_list[-1], self.op_list[-1]):]
        anomaly = HotellingT2(windowForAnomaly)
        for delete_ind, anomaly_ind in enumerate(anomaly):
            del windowForAnomaly[anomaly_ind - delete_ind] # windowForAnomaly : Anomaly excluded window 

        # Calculate offset & operation
        offset = np.mean(windowForAnomaly)
        self.operation_buffer = self.Target - offset

        # Append sub-operation index
        self.sub_op_list.append(len(self.total_data))



    def step(self, feature):
        '''
        MFB Loop
        :param feature: offset data
        :return: None
        '''
        # START
        if self.status == 'noFBapplied':
            self._put_feature(feature)

            # Enough data ?
            if len(self.total_data) - self.CPD_list[-1] < self.Nw: #FIXME index issue : DONE
                return

            # Check if lies within calculated threshold. If not, do operation.
            if self._is_within_threshold():
                # If state is on target & stable, do one sub-operation.
                if not self.sub_operation: # "False" if no sub-operation from last CPD point. (As only one sub-op applied for one ADF-test steady-state)
                    check_point = max(self.CPD_list[-1], self.op_list[-1]) #'op_list':not considered delay operation index list, 'CPD_list': BOCPD found CPD index list
                    if (len(self.total_data) - check_point) >= self.Nw:  # There must be enough data after Op. or CPD. FIXME index issue : DONE
                        if adfuller(self.total_data[check_point:])[1] < 1e-6:  # Stable?
                            self._do_sub_operation()
                return
            else:
                self._do_operation()  # SG-filter smoothing is not consider anomaly # FIXME
                self.estimator = stats.gaussian_kde(self.window)  # Set Gaussian KDE for Abrupt CPD.   # FIXME : minimum of delay given, then calculate the estimator
                self.status = 'FBapplied'
                return

        # Waiting for FB to be applied.
        if self.status == 'FBapplied':
            self._put_feature(feature)
            self.wait_num += 1 # FIXME index issue : DONE

            # Abrupt CPD : repeating low p-value point or waiting enough fires BOCD
            if self.estimator.integrate_box_1d(-np.Inf, self.total_data[-1]) < self.p or \
                    self.estimator.integrate_box_1d(-np.Inf, self.total_data[-1]) > 1 - self.p:
                self.abrupt_num += 1

            if self.wait_num >= self.Nomax:
                self._run_CPD()  # CPD fired
                self.bocd_f_ind_list.append(len(self.total_data)-1)
                self.wait_num = 0  # Reset self.wait_num
                self.abrupt_num = 0  # Reset self.abrupt_num
                self.status = 'noFBapplied'  # Reset self.status
                return

            elif self.abrupt_num >= self.Nomin:
                self._run_CPD()  # CPD fired
                self.bocd_f_ind_list.append(len(self.total_data)-1)
                self.wait_num = 0  # Reset self.wait_num
                self.abrupt_num = 0  # Reset self.abrupt_num
                self.status = 'noFBapplied'  # Reset self.status
                return

            else:
                return

    def show_param(self):
        '''
        Show parameters
        :return: None
        '''
        print('\n'+'='*40)
        print('\n>> Component :')
        print('\tArray Index : '+self.ARRAY_INDEX)
        print('\tREFERENCE : '+self.REFERENCE)
        print('\tAXIS : '+self.AXIS)
        print('\n>> Parameters :')
        print('\tNw =', self.Nw, ', Nomin =', self.Nomin, ', Nomax =', self.Nomax, ', Nstd =', self.Nstd)
        print('\tMean Delay =', self.mean_D)
        print('\tTarget Mean =', self.Target)
        print('\tOffset Threshold = 0.7 sigma')
        print('\n'+'='*40)

    def draw(self):
        '''
        Draw figures
        :return: None
        '''
        # Plot total data & operation index & cp index
        fit, (ax1) = plt.subplots(nrows=2, ncols=1, figsize=(20, 8))
        ax1[0].grid(True)
        ax1[0].set_title('Array Index : '+self.ARRAY_INDEX+'\nREFERENCE : '+self.REFERENCE+'\nAXIS : '+self.AXIS)
        for ind in self.op_list[1:]:
            ax1[0].axvline(ind, linestyle='--', color='g', linewidth=1)
        for ind in self.sub_op_list[1:]:
            ax1[0].axvline(ind, linestyle=':', color='r')
        for ind in self.cp_list[1:]:
            ax1[0].axvline(ind, linestyle=':', color='b')
        ax1[0].plot(self.total_data)
        ax1[0].axhline(0, linestyle='--', color='grey')
        legend_elements = [Line2D([0], [0], color='green', lw=2, label='Main Operation'),
                           Line2D([0], [0], color='red', lw=2, label='Sub Operation'),
                           Line2D([0], [0], color='blue', lw=2, label='After Delay')]
        ax1[0].legend(handles=legend_elements)

        # Plot total data & detected cp index
        ax1[1].plot(self.total_data)
        ax1[1].grid(True)
        ax1[1].axhline(0, linestyle='--', color='grey')
        for ind in self.op_list[1:]:
            ax1[1].axvline(ind - self.Nw, linestyle='--', color='black', linewidth=1)
        for ind in self.bocd_f_ind_list[1:]:
            ax1[1].axvline(ind, linestyle='--', color='grey', linewidth=1)
        for ind in self.CPD_list[1:]:
            ax1[1].axvline(ind, linestyle='--', color='r', linewidth=1)
        legend_elements = [Line2D([0], [0], color='red', lw=2, label='CP After Op.')]
        ax1[1].legend(handles=legend_elements)

    def _put_feature(self, feature):
        '''
        Get offset data
        :param feature: offset data
        :return: None
        '''
        # Apply operation after delay D
        if self.delay == self.D:
            self.operation += self.operation_buffer
            self.cp_list.append(len(self.total_data))

        feature += self.operation
        self.delay += 1

        # Append feature to data lists
        self.total_data.append(feature) # Collected total data
        self.window.append(feature)     # Nw corresponding window size

        # Initialize the threshold
        if len(self.total_data) == self.Nstd:   # FIXME use once at the initial stage (put outside of the state stage)
            self.offset_threshold = 0.7 * np.std(self.total_data)

        # Data smoothing
        if len(self.total_data) >= self.Nw:
            savgol_result = savgol_filter(self.total_data[-self.Nw:], 2*floor((self.Nw-1)/2)+1, 3)
            self.window_for_compare = savgol_result  ## compare threshold with sg-filter regressed line

        # Reset the threshold
        if (len(self.total_data) - self.CPD_list[-1]) % self.Nstd == 0:  ## FIXME index issue : DONE
            buffer = self.total_data[-self.Nstd:]
            if 0.7 * self.offset_threshold > np.std(buffer) or self.offset_threshold < 0.7 * np.std(buffer):
                self.offset_threshold = 0.7 * np.std(buffer)


if __name__ == '__main__':
    import time
    import pandas as pd

    # Set test data
    Data_temp = pd.read_csv("C:/Users/junho.lee/Desktop/Self_Code/GithubMFB/n1_test_500k.txt", sep=',')

    Pad_Temp = Data_temp[(Data_temp.Array_index == 1) & (Data_temp.Component_Name == 'R21')]
    PadOff_X = Pad_Temp.PAD_Length_offset / 1000.0
    PadOff_Y = Pad_Temp.PAD_Width_offset / 1000.0
    PadOff_A = Pad_Temp.PAD_Angle_offset / 1000.0

    test_data = np.array(PadOff_X[:500])

    test_loop = ToyMFB(Nw=15, Nomin=3, Nomax=20, Nstd=15)
    test_loop.show_param()

    start = time.time()  # set start time

    # Run feedback loop
    for item in test_data:
        test_loop.step(item)

    # Show RMS score before & after feedback
    RMS_before_FB = round(np.sqrt(sum((np.array(test_data) - np.zeros(len(test_data))) ** 2) / len(test_data)),2)
    RMS_after_FB = round(np.sqrt(sum((np.array(test_loop.total_data) - np.zeros(len(test_data))) ** 2) / len(test_data)),2)

    print('operation count\t: ', test_loop.operation_cnt)
    print('RMS_before_FB\t: ', RMS_before_FB)
    print('RMS_after_FB\t: ', RMS_after_FB)

    # Plot raw data & feedback result
    fit, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
    ax0.grid(True)
    plt.title('RMS before FB : ' + str(RMS_before_FB) + '\n RMS after FB : ' + str(RMS_after_FB))
    ax0.plot(test_data[:], 'grey')
    ax0.plot(test_loop.total_data[:], 'b')
    ax0.axhline(0, linestyle='--', color='grey')
    ax0.legend(['Raw Data', 'After Feedback'])

    test_loop.draw()
