class NNDummyModule1ClassForStaticAndInstanceMethod:
    def __init__(self):
        self.ke = 78

    @staticmethod
    def static_Method1():
        print('staticmethod for NNDummyModule1')

    def instanceMeth1(self):
        print('instancemethod for NNDummyModule1')
