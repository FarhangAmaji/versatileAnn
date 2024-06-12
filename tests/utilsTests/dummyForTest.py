class NNDummyFor_findClassDefinition_inADirectoryTest:
    def __init__(self):
        self.ke = 78

    @staticmethod
    def static_Method1():
        print('staticmethod for NNDummyModule1')

    def instanceMeth1(self):
        print('instancemethod for NNDummyModule1')


# ----
class DummyClassFor_isFunctionOrMethod:
    @staticmethod
    def staticMethod():
        pass

    @classmethod
    def classMethod(cls):
        pass

    def instanceMethod(self):
        pass

    def _privateMethod(self):
        pass

    def __magicMethod(self):
        pass


def dummyRegularFunctionFor_isFunctionOrMethod():
    pass
