from pycredit.utils.utils_func import _check_attrs


class TestClass:
    def __init__(self, attr1=None):
        self.attr1 = attr1

    @_check_attrs('attr1')
    def test_method(self):
        return "All attributes are set!"

if __name__ == '__main__':
    TestClass().test_method()