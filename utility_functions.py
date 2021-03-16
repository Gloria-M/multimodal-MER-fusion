def get_quadrant(measurements, value_range=(0, 1)):
    val_min, val_max = value_range
    val_mid = (val_max - val_min) / 2 + val_min

    valence, arousal = measurements
    if valence > val_mid:
        return 1 if arousal > val_mid else 4
    else:
        return 2 if arousal >= val_mid else 3


def print_epoch(epoch, train_dict, validation_dict=None):

    print_keys = {'valence_loss': 'Valence',
                  'arousal_loss': 'Arousal'}

    print('\nEPOCH {:d}'.format(epoch))
    print('-' * 35)

    print('   Train loss')
    for key, message in print_keys.items():
        print('  {:>12s} : {:.3f}'.format(message, train_dict[key][-1]))

    if validation_dict is not None:
        print('\n   Validation loss')
        for key, message in print_keys.items():
            print('  {:>12s} : {:.3f}'.format(message, validation_dict[key][-1]))


def print_test_results(valence_dict, arousal_dict, quadrants_dict):

    print('VALENCE')
    print('   MAE : {:.4f}'.format(valence_dict['mae']))
    print('   MSE : {:.4f}'.format(valence_dict['mse']))
    print()
    print('AROUSAL')
    print('   MAE : {:.4f}'.format(arousal_dict['mae']))
    print('   MSE : {:.4f}'.format(arousal_dict['mse']))
    print()
    print('Accuracy')
    for quadrant in range(1, 5):
        print('   QUADRANT {:d} : {:.2f}%%'.format(quadrant, quadrants_dict[quadrant]))


def print_params(args_dict):
    for key, val in args_dict.items():
        print(f'{key} : {val}')


def fail_format(fail_message):
    fail_flag = '===FAILED==='

    return "\n{:s}\n   {:s}\n{:s}\n".format(fail_flag, fail_message, fail_flag)


def success_format(success_message):
    success_flag = '===SUCCEEDED==='

    return "\n{:s}\n   {:s}\n{:s}\n".format(success_flag, success_message, success_flag)
