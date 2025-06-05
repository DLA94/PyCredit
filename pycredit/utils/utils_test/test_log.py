if __name__ == '__main__':
    from pycredit.utils.log import Logger
    import logging

    logger = Logger(level=logging.DEBUG)

    @logger
    def add(a, b):
        return a + b

    add(1, 2)
    logger.log('info', 'test msg')
    logger.log('error', 'test msg')
    logger.log('warning', 'test msg')
    logger.log('critical', 'test msg')
    logger.log('debug', 'test msg')