# phase1: normal regression
batchsize 

maybe make notBetterScoreCounter to reduce learningRate

l1 and l2 regularizations in general and due to layer

patience

save model on memory and have option to save on hardDrive on epoch

check save path should be provided

do parallel data preparation

may add data preparation with outlier dectection

# phase2: modes
depending on classification, binary classification make final forward step, and datashape check in train

modes:
    binary classification has sigmoid in last layer
    classification has softmax in last layer
    binary classification should have 1 final outputSize if has 2 we use regular classification
    classification should have multidim(more than 1) or we can change it to onehot
    classifications may use accuracy or loss for their modelEval
    dropoutEnsemble get numSamples
    variationalEncoder must get latentDim, in args
    variationalEncoder must have reparameterize and klDivergenceLoss

check dropout ensemble to do dropout in eval

hyperParam tuning with mopso(is it possible)


# done
make a learning rate change func

kkk should optimizer be in init
#%% implement the post init later
class class00:
    def __init__(self):
        print('class00 __init__')

    def common2(self):
        print('common 2')


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class BaseClass(class00, metaclass=PostInitCaller):

    def __init__(self):
        super().__init__()
        print('base __init__')
        self.common1()

    def common1(self):
        print('common 1')

    def finalizeInitialization(self):
        print('finalizeInitialization [common2]')

    def __post_init__(self):  # this is called at the end of __init__
        self.finalizeInitialization()


class Subclass1(BaseClass):
    def __init__(self):
        super().__init__()
        self.specific()

    def specific(self):
        print('specific')


s = Subclass1()
