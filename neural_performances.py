from data_import import dict_import, select_data, paths, exceptions, specific_time, dates, normalize_data
from utilities import train, ts2windows, unscalerise, nilm_f1score, def_model
from sklearn.metrics import mean_absolute_error
from data_format import window_format_val
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

#########################################################
exp_title = 'testv5_model_full'
sample_s = 7
batchsz = 512
epochs = 12
patience = 6
#########################################################
APPLIANCES = ['washing machine', 'television', 'dish washer', 'computer', 'microwave', 'fridge freezer',  'kettle',
              'freezer']
all_seqlen = [102, 102, 171, 102, 102, 102, 85, 102]

#APPLIANCES = ['washing machine', 'television']
#all_seqlen = [102, 102]
#########################################################
mode = 'test'  # 'train' or 'test'
test_mode = 'select'  # mode 'select' for a limited number of houses, else 'auto' or 'select_random'
mode_norm = 'standard'  # mode for choosing scaler: 'standard', 'min_max' or 'unscale'
neural_mode = 'reduced'  # normal or reduced for neural model
#########################################################
nb = 10  # number of max random chosen houses
weeks = 2
n_houses_trained = 7
t = 6000 + 15000 + 4000  # graph display parameters
thr = 100  # f1 score threshold
delta = 1000
save = 'OFF'  # 'ON' for saving graph, else 'OFF' => create directory file before !
#########################################################

# format files and timestamps
experience_title, folder_location, graph_loc, REFIT_path = paths(exp_title)
reference, APPLIANCES_dict, ALL_APPLIANCES = dict_import(REFIT_path)
reference, APPLIANCES_dict = exceptions(reference, APPLIANCES_dict)  # remove corrupted data from the experience
dates = dates()  # gather all the selected dates
time_periods = specific_time(dates, reference['buildings_ids'], weeks=weeks)  # or use the function tim_periods
print(ALL_APPLIANCES)

# importing raw data
import_kwargs, train_kwargs = {'APPLIANCES_dict': APPLIANCES_dict, 'apps': APPLIANCES, 'path': REFIT_path,
                               'sample_s': sample_s, 'time_periods': time_periods, 'mode': test_mode,
                               'house_number': nb}, {'batchsz': batchsz, 'epochs': epochs, 'patience': patience, }
seqlens, test_kwargs, m = {}, {}, 0

data, bd_ids = select_data(**import_kwargs)
for k, appliance in enumerate(APPLIANCES):
    seqlens[appliance] = all_seqlen[k]

# correct data errors and normalize it
data, scalers = normalize_data(data, mode_norm)
outputs = {}
for j in range(n_houses_trained):
    outputs[j+1] = {}

true_test_kwargs = {}
for appliance in APPLIANCES:
    i = 1
    assert mode in ['test', 'train']
    seqlen = seqlens[appliance]

    houses = list(data[appliance])
    test_kwargs = {houses[-1]: data[appliance][houses[-1]], houses[-2]: data[appliance][houses[-2]],
                   houses[-3]: data[appliance][houses[-3]], 'timestamps': {}}
    test_kwargs['timestamps'][houses[-1]] = data[appliance][houses[-1]][appliance].index
    test_kwargs['timestamps'][houses[-2]] = data[appliance][houses[-2]][appliance].index
    test_kwargs['timestamps'][houses[-3]] = data[appliance][houses[-3]][appliance].index

    for n in range(n_houses_trained):
        n_houses = n+1
        model = def_model(mode, neural_mode, seqlen, folder_location, experience_title, appliance, n_houses)
        train_kwargs['model'] = model

        outputs[n_houses][appliance] = {}
        # for future usage
        true_test_kwargs[appliance] = test_kwargs

        window_format_val(data, appliance, train_kwargs, seqlen)

        # train & save model
        if mode == 'train':
            history = train(**train_kwargs)
            model.save(folder_location + experience_title + '/' + appliance)

        for house in list(test_kwargs.keys())[:-1]:
            # testing on an unknown house
            test = test_kwargs[house]
            output = model.predict(ts2windows(test['mains'].values, seqlen, padding='output'))
            unscaled_values = unscalerise(test, house, appliance, 'known', scalers)
            unscaled_values_output = unscalerise(test, house, appliance, 'output', scalers, output)

            outputs[n_houses][appliance][house] = {'output': output, 'unscaled_values': unscaled_values,
                                                   'unscaled_values_output': unscaled_values_output}

            # score calculation, exception with television
            if appliance == 'computer':
                thr_ = 10
            elif appliance == 'fridge freezer':
                trh_ = 10
            else:
                thr_ = thr
            outputs[n_houses][appliance][house]['f1'] = \
                nilm_f1score(unscaled_values[appliance], unscaled_values_output[appliance], thr_)

            outputs[n_houses][appliance][house]['mae'] = \
                mean_absolute_error(unscaled_values[appliance], unscaled_values_output[appliance])
            print(str(i) + ':' + str(len(APPLIANCES)) + ' done')
        i = i+1

# outputs[n][appliance]

f1s, maes = {}, {}
for n in range(n_houses_trained):
    n_houses = n + 1
    f1s[n_houses], maes[n_houses] = {}, {}

    for appliance in APPLIANCES:
        f1s[n_houses][appliance] = []
        maes[n_houses][appliance] = []
        for house in outputs[n_houses][appliance].keys():

            f1s[n_houses][appliance].append(outputs[n_houses][appliance][house]['f1'])
            maes[n_houses][appliance].append(outputs[n_houses][appliance][house]['mae'])

    # moyennes
        f1s[n_houses][appliance] = sum(f1s[n_houses][appliance]) / len(f1s[n_houses][appliance])
        maes[n_houses][appliance] = sum(maes[n_houses][appliance]) / len(maes[n_houses][appliance])

'''
    normal graph
                '''
x = [x+1 for x in range(n_houses_trained)]
for appliance in APPLIANCES:
    f = [f1s[n+1][appliance] for n in range(n_houses_trained)]
    plt.plot(x, f)
plt.legend(APPLIANCES)
plt.xlabel('Number of trained buildings')
plt.ylabel('f1 scores')
plt.title('thr = 10 for computer and fridge freezer')
plt.show()
for appliance in APPLIANCES:
    m = [maes[n + 1][appliance] for n in range(n_houses_trained)]
    plt.plot(x, np.array(m)/100)
plt.legend(APPLIANCES)
plt.xlabel('Number of trained buildings')
plt.ylabel('MAE (100 W)')
plt.show()

'''
    smooth graph
                '''
x = [x + 1 for x in range(n_houses_trained)]
for appliance in APPLIANCES:
    f = [f1s[n+1][appliance] for n in range(n_houses_trained)]
    x_ = np.array(x)
    y = np.array(f)
    X_Y_Spline = make_interp_spline(x_, y)
    X_ = np.linspace(x_.min(), x_.max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_)
plt.legend(APPLIANCES)
plt.xlabel('Number of trained buildings')
plt.ylabel('f1 scores')
plt.show()
for appliance in APPLIANCES:
    m = [maes[n + 1][appliance] for n in range(n_houses_trained)]
    x_ = np.array(x)
    y = np.array(m)
    X_Y_Spline = make_interp_spline(x_, y)
    X_ = np.linspace(x_.min(), x_.max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_)
plt.legend(APPLIANCES)
plt.xlabel('Number of trained buildings')
plt.ylabel('MAE (100 W)')
plt.show()

####
##########
'''change threshold for television'''
'''test for model == FULL'''
##########
####

t = 500
dt = 1000
plt.plot(np.array(outputs[7]['washing machine']['building10']['unscaled_values']['mains'])[t:t+dt])
plt.plot(np.array(outputs[7]['washing machine']['building10']['unscaled_values_output']['washing machine'])[t:t+dt])
plt.show()
