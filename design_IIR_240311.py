

#https://aidiary.hatenablog.com/entry/20120103/1325594723

#coding:utf-8
import wave
import struct
import numpy as np
from pylab import *
import pandas as pd

"""IIRフィルタ"""

def createLPF(fc):
    """IIR版ローパスフィルタ、fc:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2
    b[0] = (4 * np.pi**2 * fc**2) / denom
    b[1] = (8 * np.pi**2 * fc**2) / denom
    b[2] = (4 * np.pi**2 * fc**2) / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc**2 - 2) / denom
    a[2] = (1 - (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2) / denom
    return a, b

def createHPF(fc):
    """IIR版ハイパスフィルタ、fc:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2
    b[0] = 1.0 / denom
    b[1] = -2.0 / denom
    b[2] = 1.0 / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc**2 - 2) / denom
    a[2] = (1 - (2 * np.sqrt(2) * np.pi * fc) + 4 * np.pi**2 * fc**2) / denom
    return a, b

def createBPF(fc1, fc2):
    """IIR版バンドパスフィルタ、fc1、fc2:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2
    b[0] = (2 * np.pi * (fc2 - fc1)) / denom
    b[1] = 0.0
    b[2] = - 2 * np.pi * (fc2 - fc1) / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc1 * fc2 - 2) / denom
    a[2] = (1.0 - 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2) / denom
    return a, b

def createBSF(fc1, fc2):
    """IIR版バンドストップフィルタ、fc1、fc2:カットオフ周波数"""
    a = [0.0] * 3
    b = [0.0] * 3
    denom = 1 + 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2
    b[0] = (4 * np.pi**2 * fc1 * fc2 + 1) / denom
    b[1] = (8 * np.pi**2 * fc1 * fc2 - 2) / denom
    b[2] = (4 * np.pi**2 * fc1 * fc2 + 1) / denom
    a[0] = 1.0
    a[1] = (8 * np.pi**2 * fc1 * fc2 - 2) / denom
    a[2] = (1 - 2 * np.pi * (fc2 - fc1) + 4 * np.pi**2 * fc1 * fc2) / denom
    return a, b

# まだ未完成(うまく動かない。多分計算ミス)
# def create_1D_BSF(fc1, fc2):
#     """IIR版バンドストップフィルタ、fc1、fc2:カットオフ周波数"""
#     a = [0.0] * 3 # 係数a
#     b = [0.0] * 3 # 係数b
#     denom = 1 + 4 * np.pi ** 2 * fc2 * fc1 + 2 * np.pi * (fc2 - fc1) # 分母
#     b[0] = (4 * np.pi * fc2 * fc1 + 1) / denom
#     b[1] = (8 * np.pi * fc2 * fc1 - 2) / denom
#     b[2] = (4 * np.pi * fc2 * fc1 + 1) / denom
#     a[0] = 1.0
#     a[1] = (8 * np.pi ** 2 * fc1 * fc2 -2) / denom
#     a[2] = (4 * np.pi ** 2 * fc1 * fc2 -2 * np.pi * (fc2 - fc1)) / denom
    
#     return a, b



def iir(x, a, b):
    """IIRフィルタをかける、x:入力信号、a, b:フィルタ係数"""
    y = [0.0] * len(x)  # フィルタの出力信号

    Q = len(a) - 1
    P = len(b) - 1
    for n in range(len(x)):
        for i in range(0, P + 1):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        for j in range(1, Q + 1):
            if n - j >= 0:
                y[n] -= a[j] * y[n - j]
        # print(y[n])
    return y



import scipy.fftpack as fft



def my_freqz(b, a=[1], worN=None):
    lastpoint = np.pi
    N = 512 if worN is None else worN
    w = np.linspace(0.0, lastpoint, N, endpoint=False)
    h = fft.fft(b, 2 * N)[:N] / fft.fft(a, 2 * N)[:N]
    return w, h



def generate_sine_wave(frequency, duration, sampling_rate):
    # 時間軸の生成
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # 正弦波の生成
    x = np.sin(2 * np.pi * frequency * t)

    return x, t


def generate_square_wave(frequency, duration, sampling_rate):
    # 時間軸の生成
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # 1周期の時間
    period = 1 / frequency

    # 矩形波の生成
    x = np.where((t % period) < period / 2, 1, -1)

    return x, t



# フィルター関数は省略
if __name__ == '__main__':

    fs = 5000 # サンプリング周波数
    # fs = float(input("サンプリングレートを入力: ")) # サンプリング周波数
    
    # 2. 設計するフィルタの指定
    # filter_type = input("フィルタの種類(LPF, HPF, BPF, BSF)を入力: ")
    filter_type = "BSF"

    if filter_type == "LPF":
        fc_digital = float(input("カットオフ周波数を入力: "))
        fc_analog = np.tan(fc_digital*np.pi/fs)/(2*np.pi)
        a, b = createLPF(fc_analog)
    elif filter_type == "HPF":
        fc_digital = float(input("カットオフ周波数を入力: "))
        fc_analog = np.tan(fc_digital*np.pi/fs)/(2*np.pi)
        a, b = createHPF(fc_analog)
    elif filter_type == "BPF":
        fc1_digital = float(input("カットオフ周波数下限fc1を入力: "))
        fc2_digital = float(input("カットオフ周波数下限fc2を入力: "))
        fc1_analog = np.tan(fc1_digital*np.pi/fs)/(2*np.pi)
        fc2_analog = np.tan(fc2_digital*np.pi/fs)/(2*np.pi)
        a, b = createBPF(fc1_analog, fc2_analog)
    elif filter_type == "BSF":
        # fc1_digital = float(input("カットオフ周波数下限fc1を入力: "))
        fc1_digital = 840
        fc2_digital = float(input("カットオフ周波数下限fc2を入力: "))
        fc1_analog = np.tan(fc1_digital*np.pi/fs)/(2*np.pi)
        fc2_analog = np.tan(fc2_digital*np.pi/fs)/(2*np.pi)
        a, b = createBSF(fc1_analog, fc2_analog)
    else:
        print("フィルタの種類が不正です")
        exit()


    # 入力用正弦波生成
    # input_signal_frequency = 50  # 周波数 (Hz)
    # input_signal_frequency = float(input("テスト用正弦波の周波数: "))  # 周波数 (Hz)
    input_signal_frequency = 851


    input_signal_duration = 5  # 長さ (seconds)

    # fc1_digital = 49.5
    # fc2_digital = 50.5
    # fc1_digital = 47
    # fc2_digital = 53

    # fc1_analog = np.tan(fc1_digital*np.pi/fs)/(2*np.pi)
    # fc2_analog = np.tan(fc2_digital*np.pi/fs)/(2*np.pi)
    # a, b = createBSF(fc1_analog, fc2_analog)

    print("a: " + str(a))
    print("b: " + str(b))

  # 周波数特性の計算
    # w, h = sg.freqz(b, a)

    # w, h = my_freqz(b, a, 2**24);
    w, h = my_freqz(b, a, 2**24);








    input_signal, t = generate_sine_wave(input_signal_frequency, input_signal_duration, fs)
    # input_signal, t = generate_square_wave(input_signal_frequency, input_signal_duration, fs)

    
    
    
     # フィルタをかける
    output_signal = iir(input_signal, a, b)
    


    # rad/sampleをHzに変換
    w_hz = w / (2 * np.pi) * fs  


    # 周波数特性のプロット1
    fig = plt.figure()
    ax = fig.add_subplot(311) # 1行1列の描画領域を確保
    ax.plot(w, 20 * np.log10(np.abs(h)), c="red")
    ax.set_ylim([-100, 5]) 
    ax.set_xlabel("Frequency [rad/sample]")
    ax.set_ylabel("Amplitude [dB]")
    


    plt.title("Frequency Response")
    plt.grid()



    # 周波数特性のプロット2
    ax2 = ax = fig.add_subplot(312)
    # ax2 = ax.twinx() 
    ax2.set_xlim([845,855]) #x軸の範囲を0~200に指定
    ax2.set_ylim([-100,5]) #y軸の範囲を-50~5に指定
    ax2.plot(w_hz, 20 * np.log10(np.abs(h)), c="blue")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [dB]", color='b')
    ax2.tick_params('y', colors='b')



    fig.tight_layout()
    # plt.show()


    # 波形のプロット
    ax3 = fig.add_subplot(313)
    # ax3.set_xlim([4, 4.1]) #x軸の範囲を4~4.1に指定
    ax3.set_xlim([0, 4.1]) #x軸の範囲を4~4.1に指定
    ax3.plot(t, input_signal, c="green", label="input signal")
    ax3.plot(t, output_signal, c="red", label="output signal")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Amplitude")
    ax3.legend()




    plt.title("Input Signal")
    plt.grid()

    



    plt.show()




