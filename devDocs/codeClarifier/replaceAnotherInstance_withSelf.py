"""
note this code is not successful and can't replace another instance with self
"""


class A:
    def __init__(self, a):
        self.a = a

    def replace(self, obj):
        self = obj


ins1 = A(10)
ins2 = A(20)
ins1.replace(ins2)
