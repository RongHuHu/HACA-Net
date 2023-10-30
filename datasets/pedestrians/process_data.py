import sys
import os
import numpy as np
import pandas as pd
import dill

sys.path.append("../../trajectron")
from environment import Environment, Scene, Node
from utils import maybe_makedirs
from environment import derivative_of

desired_max_time = 100#将100赋值给变量desired_max_time，表示所需的最大时间。
pred_indices = [2, 3]#定义一个包含2和3的列表pred_indices，表示所需的预测索引
state_dim = 6#将6赋值给变量state_dim，表示状态维度为6
frame_diff = 10#将10赋值给变量frame_diff，表示帧之间的差异为10
desired_frame_diff = 1#将1赋值给变量desired_frame_diff，表示所需的帧之间的差异为1
dt = 0.4#将0.4赋值给变量dt，表示时间差为0.4
#接下来定义了一个名为standardization的字典，该字典包含了标准化数据的均值和标准差，用于将数据标准化到统一的范围内。该字典的结构如下
standardization = {
    'PEDESTRIAN': {         #键为PEDESTRIAN，表示这个标准化数据适用于行人
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}


def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay}

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0
maybe_makedirs('../processed')#将在目录 '../processed' 中创建一个新目录（如果该目录不存在）
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
#此代码行使用 pandas 库中的 MultiIndex 方法，创建了一个具有两层索引的多重索引对象。该对象的第一层包含三个元素 'position'、'velocity' 和 'acceleration'，第二层包含两个元素 'x' 和 'y'。
for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:#将 desired_source 变量设置为 eth、hotel、univ、zara1 和 zara2 中的一个
    for data_class in ['train', 'val', 'test']:#将 data_class 变量设置为 train、val 或 test 中的一个。这两个循环组合在一起，可以迭代处理所有可能的组合。
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)#将一个参数 node_type_list 设置为一个包含一个元素 'PEDESTRIAN' 的列表，并从另一个变量 standardization 中获取一个值
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius
#创建了一个名为 attention_radius 的空字典，然后向该字典中添加了一个键值对 (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0，最后将 attention_radius 赋值给 env.attention_radius
        scenes = []
        data_dict_path = os.path.join('../processed', '_'.join([desired_source, data_class]) + '.pkl')
#使用 os 库中的 join 方法创建了一个名为 data_dict_path 的字符串，该字符串将字符串 '../processed'、desired_source、data_class 和字符串 '.pkl' 连接在一起
        for subdir, dirs, files in os.walk(os.path.join('raw', desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)
#这几行代码首先使用 os 库中的 walk 函数遍历指定目录（即 'raw' + desired_source + data_class）中的所有文件和文件夹，然后在每个文件中查找以 '.txt' 结尾的文件。
# 如果找到了符合条件的文件，则创建一个名为 input_data_dict 的空字典，并使用 os 库中的 join 方法创建一个名为 full_data_path 的字符串，该字符串将子目录、文件名和目录路径连接在一起。
# 然后，代码打印出 full_data_path。
                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)#从指定路径的文件中读取数据，并以'\t'为分隔符，不使用文件中的列名，所以header参数设置为None。
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']#使用data.columns指定列名
                    #print('data1',data)
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')#将frame_id和track_id列的数据类型都转换为整型
                    #print('data2',data)
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
                    #print('data3',data)
                    data['frame_id'] = data['frame_id'] // 10#将frame_id列的数据每隔10个取一个，相当于将数据的时间步长变为了原来的1/10，这样做是为了减少数据量，降低计算难度
                    #print('data4',data)
                    data['frame_id'] -= data['frame_id'].min()#通过减去frame_id列中的最小值，将时间步长的起始点从0开始，而不是从一个大的负数开始
                    #print('data5',data)
                    data['node_type'] = 'PEDESTRIAN'#为每一个数据点指定节点类型为'PEDESTRIAN'，并将track_id转换为字符串类型并赋值给node_id
                    #print('data6',data)
                    data['node_id'] = data['track_id'].astype(str)
                    #print('data7',data)
                    data.sort_values('frame_id', inplace=True)#对数据按frame_id进行排序
                    #print('data8',data)
                    # Mean Position
                    data['pos_x'] = data['pos_x'] - data['pos_x'].mean()#计算x和y方向的坐标平均值，并将每个数据点的坐标减去对应方向的坐标平均值，从而将数据的坐标中心移动到原点
                    #print('data9',data)
                    data['pos_y'] = data['pos_y'] - data['pos_y'].mean()
                    #print('data10',data)
                    max_timesteps = data['frame_id'].max()#计算数据集中的最大时间步长，并将其保存到max_timesteps变量中
                    #print('max_timesteps',max_timesteps)
                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=augment if data_class == 'train' else None)
#创建一个名为scene的Scene对象，其中timesteps的值为max_timesteps+1，dt表示时间步长，name为数据源和数据集类型的组合，aug_func是用于数据增强的函数。如果当前处理的是train集，则aug_func为augment函数，否则为None。
                    for node_id in pd.unique(data['node_id']):#获取数据集 data 中所有不同的 node_id 值
                    
                        node_df = data[data['node_id'] == node_id]#获取 data 中 node_id 值为当前循环的 node_id 的所有行组成的 DataFrame
                        #print('node_df',node_df)
                        assert np.all(np.diff(node_df['frame_id']) == 1)#断言当前 node_df 的 frame_id 列中的值都是连续的，即每个 frame_id 的值都比前一个值大 1，如果不满足这个条件就会触发 AssertionError

                        node_values = node_df[['pos_x', 'pos_y']].values#从 node_df 中获取 pos_x 和 pos_y 列的值组成的 NumPy 数组 node_values
                        #print('node_values',node_values)
                        if node_values.shape[0] < 2:#如果 node_values 数组的行数小于 2，即该节点数据不完整，跳过后续处理
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0]#获取 node_df 的第一行的 frame_id 值，作为该节点的第一个时间步

                        x = node_values[:, 0]#从 node_values 数组中获取第 1 列的值，即节点在每个时间步的 x 坐标值
                        y = node_values[:, 1]#从 node_values 数组中获取第 2 列的值，即节点在每个时间步的 y 坐标值
                        vx = derivative_of(x, scene.dt)#计算节点在每个时间步的 x 方向速度，这里使用了一个名为 derivative_of 的函数
                        vy = derivative_of(y, scene.dt)#计算节点在每个时间步的 y 方向速度，同样使用了 derivative_of 函数
                        ax = derivative_of(vx, scene.dt)#计算节点在每个时间步的 x 方向加速度，同样使用了 derivative_of 函数
                        ay = derivative_of(vy, scene.dt)#计算节点在每个时间步的 y 方向加速度，同样使用了 derivative_of 函数

                        data_dict = {('position', 'x'): x,
                                     ('position', 'y'): y,
                                     ('velocity', 'x'): vx,
                                     ('velocity', 'y'): vy,
                                     ('acceleration', 'x'): ax,
                                     ('acceleration', 'y'): ay}
                        #print('data_dict',data_dict)
#将 x、y、vx、vy、ax、ay 这 6 个 NumPy 数组打包成一个字典 data_dict，字典的键是一个元组，元组的第一个元素是 'position'、'velocity' 或 'acceleration'，
# 表示这个数据是位置、速度还是加速度；元组的第二个元素是 'x' 或 'y'，表示这个数据是 x 轴还是 y 轴上的值
                        node_data = pd.DataFrame(data_dict, columns=data_columns)#用 data_dict 字典中的数据创建一个 Pandas DataFram
                        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                        node.first_timestep = new_first_idx

                        scene.nodes.append(node)
                    if data_class == 'train':
                        scene.augmented = list()
                        angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    print(scene)
                    scenes.append(scene)
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

print(f"Linear: {l}")
print(f"Non-Linear: {nl}")