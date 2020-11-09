import numpy

def reader_creator_random_image(width, height):
    '''
    单项目数据读取器创建者
    '''

    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height)
    return reader

def reader_creator_random_image_and_label(width, height, label):
    '''
    多项目数据读取器创建者
    '''

    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height), label
    return reader

# def pipe_reader():
#     for f in ['asset/a.txt','asset/b.txt']:
#         pr = PipeReader("cat %s"%f)
#         for l in pr.get_line():
#             sample = l.split(" ")
#             yield sample

# pipe_reader()