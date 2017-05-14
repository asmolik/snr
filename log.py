""" Logging stuff """

def message(msg):
    ''' Logs message '''
    print(msg)

def dataset_configuration(feature, classes_cnt, components_cnt):
    ''' Logs dataset configuration info'''
    print('Selected feature: {}'.format(feature))
    print('Classes count: {:d}'.format(classes_cnt))
    print('Principal components count: {:d}'.format(components_cnt))

def dataset_summary(train_set_size, test_set_size):
    ''' Logs dataset summary '''
    full_dataset_size = train_set_size + test_set_size
    print('Full dataset size: {:d}'.format(full_dataset_size))
    print('Training set size: {:d} ({:.2f} %)'.format(
        train_set_size,
        100*train_set_size/full_dataset_size
    ))
    print('Test set size: {:d} ({:.2f} %)'.format(
        test_set_size,
        100*test_set_size/full_dataset_size
    ))
    print('')

def time(elapsed_seconds):
    ''' Logs operation elapsed time '''
    total_seconds = int(elapsed_seconds)
    hours = total_seconds//3600
    minutes = (total_seconds - 3600*hours)//60
    seconds = total_seconds - 3600*hours - 60*minutes
    msg = "Operation completed. Elapsed time: {:d}h {:d}m {:d}s".format(hours, minutes, seconds)
    print(msg)

def empty_line():
    ''' Logs empty line '''
    print('')
