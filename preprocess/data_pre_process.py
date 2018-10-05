from preprocess.pattern_encoder import encode_column_to_range_index, decode_column_to_int

from ta import *


def process_change_series(close_s, step_back_s):
    series = (step_back_s - close_s) * 100 / close_s
    # print(series[-1:])
    return series.apply(encode_column_to_range_index)


def create_data_frame(data_csv, considering_steps=15,
                      rsi_range=[14, 29, 58, 100],
                      tsi_range=[14, 29, 58, 100],
                      emi_range=[9, 11, 20, 100],
                      aroon_range=[9, 13, 29, 50],
                      dpo_range=[4, 5, 13, 35]):
    data_csv['Ct'] = data_csv.Close.shift(considering_steps)
    data_csv.dropna(inplace=True)
    # print(data_csv.head())
    df = pd.DataFrame()
    close_s = data_csv.Close
    df['C'] = close_s

    for rsi_i in rsi_range:
        df['RSI({})'.format(rsi_i)] = rsi(close_s) / 100

    for atr_i in tsi_range:
        df['ATR({})'.format(atr_i)] = average_true_range(data_csv.High, data_csv.Low, close_s, n=atr_i)

    for ema_i in emi_range:
        df['exp({})'.format(ema_i)] = ema(close_s, ema_i)

    for aron_i in aroon_range:
        df['arn_d({})'.format(aron_i)] = aroon_down(close_s, n=aron_i)
        df['arn_u({})'.format(aron_i)] = aroon_up(close_s, n=aron_i)
    #
    for dpo_i in dpo_range:
        df['dpo({})'.format(dpo_i)] = ema(data_csv.Close, dpo_i)

    # Pattern
    series = (close_s.shift(1) - close_s) * 100 / close_s
    series = series.apply(encode_column_to_range_index)
    df['P1'] = series
    df['P2'] = series
    df['P3'] = series
    df['P4'] = series
    #
    for back_step in range(2, (considering_steps - 1) + 1):
        df['P1'] += process_change_series(close_s, close_s.shift(back_step))

    #
    for back_step in range(2, 5):
        df['P2'] += process_change_series(close_s, close_s.shift(back_step))

    #
    for back_step in range(2, 4):
        df['P3'] += process_change_series(close_s, close_s.shift(back_step))

    for back_step in range(2, 10):
        df['P4'] += process_change_series(close_s, close_s.shift(back_step))
    # print(df['P'])
    df.dropna(inplace=True)
    # print(df.values[-10:, -4:])
    df['P1'] = df.P1.apply(decode_column_to_int)
    df['P2'] = df.P2.apply(decode_column_to_int)
    df['P3'] = df.P3.apply(decode_column_to_int)
    df['P4'] = df.P4.apply(decode_column_to_int)

    return df
