import numpy as np

def batch_iter(source_data,batch_size,num_epochs,shuffle = True):
    # 参考自网上https://blog.csdn.net/appleml/article/details/57413615，原出处不知
    # 数据source_data的next_batch方法，source_data为包含标签以及数据的数组
    # 返回为包含batch_size个数量的样本的数组
    # 使用yeild关键字实现，调用时使用next方法即可得到
    # 待解决问题：每个epoch如果取最后一个batch_size会报错，所以num_batches_per_epoch数量取下整
    data_size = len(source_data)
    num_batches_per_epoch = int(len(source_data) / batch_size)
    for _ in range(num_epochs):
        # 每个epoch时混洗数组
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = source_data[shuffle_indices]
        else:
            shuffled_data = source_data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]