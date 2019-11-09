# import standard modules

# import custom modules


def getSetupDicts(n):
    """
    Choose from some stock setupDicts.
    More for showing examples than being practically useful,
    as n doesn't follow any real meaningful logic
    """
    n1 = {
        'verbose': True,
        'trapezoidalFlag': False,
        'ZLiveType': 'average X',
        'terminationType': 'evidence',
        'terminationFactor': 0.2,
        'sampler': 'MH',
        'outputFile': '../text_output/test',
        'space': 'linear'}
    n2 = {
        'verbose': True,
        'trapezoidalFlag': False,
        'ZLiveType': 'average X',
        'terminationType': 'evidence',
        'terminationFactor': 0.1,
        'sampler': 'MH',
        'outputFile': '../text_output/test',
        'space': 'log'}
    n3 = {
        'verbose': True,
        'trapezoidalFlag': False,
        'ZLiveType': 'average X',
        'terminationType': 'evidence',
        'terminationFactor': 0.01,
        'sampler': 'blind',
        'outputFile': '../text_output/test',
        'space': 'linear'}
    n4 = {
        'verbose': True,
        'trapezoidalFlag': True,
        'ZLiveType': 'average Lhood',
        'terminationType': 'evidence',
        'terminationFactor': 0.2,
        'sampler': 'MH',
        'outputFile': '../text_output/test',
        'space': 'linear'}
    n5 = {
        'verbose': True,
        'trapezoidalFlag': False,
        'ZLiveType': 'max Lhood',
        'terminationType': 'evidence',
        'terminationFactor': 0.2,
        'sampler': 'MH',
        'outputFile': '../text_output/test',
        'space': 'linear'}
    n6 = {
        'verbose': True,
        'trapezoidalFlag': False,
        'ZLiveType': 'average X',
        'terminationType': 'information',
        'terminationFactor': 2.,
        'sampler': 'MH',
        'outputFile': '../text_output/test',
        'space': 'linear'}
    SDDict = {'n1': n1, 'n2': n2, 'n3': n3, 'n4': n4, 'n5': n5, 'n6': n6}
    setupDict = SDDict[n]
    return setupDict
